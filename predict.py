import os
import shutil
import random
import json
from typing import List
from cog import BasePredictor, Input, Path
from helpers.comfyui import ComfyUI

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"

with open("tshirt-api.json", "r") as file:
    workflow_json = file.read()


class Predictor(BasePredictor):
    def setup(self):
        logo_path = "replicate-logo.png"
        target_path = os.path.join(INPUT_DIR, logo_path)
        os.makedirs(INPUT_DIR, exist_ok=True)
        if not os.path.exists(target_path):
            shutil.copy(logo_path, INPUT_DIR)

        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)
        self.comfyUI.load_workflow(workflow_json)

    def cleanup(self):
        self.comfyUI.clear_queue()
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)
        os.makedirs(OUTPUT_DIR)

    def log_and_collect_files(self, directory, prefix=""):
        files = []
        for f in os.listdir(directory):
            if f == "__MACOSX":
                continue
            path = os.path.join(directory, f)
            if os.path.isfile(path):
                print(f"{prefix}{f}")
                files.append(Path(path))
            elif os.path.isdir(path):
                print(f"{prefix}{f}/")
                files.extend(self.log_and_collect_files(path, prefix=f"{prefix}{f}/"))
        return files

    def update_workflow(self, workflow, **kwargs):
        prompt = workflow["69"]["inputs"]
        prompt["text"] = (
            f"TshirtDesignAF, T shirt design, {kwargs['prompt']}, thick outline"
        )

        negative_prompt = workflow["64"]["inputs"]
        negative_prompt["text"] = (
            f"nsfw, nude, {kwargs['negative_prompt']}, photo, photography"
        )

        apply_controlnet = workflow["66"]["inputs"]
        apply_controlnet["strength"] = kwargs["control_illusion_strength"]

        sampler = workflow["2"]["inputs"]
        sampler["seed"] = kwargs["seed"]

        empty_latent_image = workflow["65"]["inputs"]
        empty_latent_image["batch_size"] = kwargs["number_of_images"]

    def predict(
        self,
        prompt: str = Input(default="a forest"),
        negative_prompt: str = Input(
            default="",
            description="Things you do not want in the image",
        ),
        control_illusion_strength: float = Input(
            default=1.35,
            ge=0,
            le=3,
            description="Strength of logo. The bigger this is, the more the logo will be applied.",
        ),
        number_of_images: int = Input(
            default=1, ge=1, le=10, description="Number of images to generate"
        ),
        seed: int = Input(
            default=None, description="Fix the random seed for reproducibility"
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        self.cleanup()

        if seed is None:
            seed = random.randint(0, 2**32 - 1)
            print(f"Random seed set to: {seed}")

        workflow = json.loads(workflow_json)
        self.update_workflow(
            workflow,
            seed=seed,
            prompt=prompt,
            negative_prompt=negative_prompt,
            control_illusion_strength=control_illusion_strength,
            number_of_images=number_of_images,
        )

        wf = self.comfyUI.load_workflow(workflow, check_weights=False)
        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)

        files = []
        print(f"Contents of {OUTPUT_DIR}:")
        files.extend(self.log_and_collect_files(OUTPUT_DIR))

        print(files)

        return files
