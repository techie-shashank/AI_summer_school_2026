from __future__ import annotations

import base64
import os
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import cv2
from dotenv import load_dotenv

from preprocess import CODE_LENGTH
from llm.common_llm import call_llm_on_fly

load_dotenv()

DEFAULT_KKY_OLLAMA_SERVER = "https://ollama.kky.zcu.cz"


DEFAULT_PROMPT = (
	"You are an expert OCR system reading numerical codes stamped into "
	f"industrial security seals. Each seal shows exactly one {CODE_LENGTH}-digit "
	"code made only of the digits 0-9. Read the digits in the exact order they "
	"appear, left to right, even if the seal is scratched, dirty, or glares. "
	"Ignore any other numbers, letters, or text printed near the code."
)

DEFAULT_COMMAND = (
	f"Report the exact {CODE_LENGTH}-digit numerical code stamped on the seal "
	"in the attached image."
)


def encode_image(image_path: str | Path, grayscale: bool = False) -> str:
	"""Load an image with OpenCV and return it as a base64-encoded PNG string."""
	flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
	image = cv2.imread(str(image_path), flag)
	if image is None:
		raise FileNotFoundError(f"Could not read image: {image_path}")
	success, buffer = cv2.imencode(".png", image)
	if not success:
		raise ValueError(f"Could not encode image: {image_path}")
	return base64.b64encode(buffer).decode("utf-8")


def is_plausible_code(code: str, length: int = CODE_LENGTH) -> bool:
	"""A valid seal code is exactly `length` digits, nothing else."""
	return bool(code) and code.isdigit() and len(code) == length


class SealCodeVLM:
	"""Predicts a seal's numerical code by asking a vision-language model to read it."""

	def __init__(
		self,
		openai: bool = True,
		model: str = "gemma3:12b",
		prompt: str = DEFAULT_PROMPT,
		command: str = DEFAULT_COMMAND,
		grayscale: bool = False,
		openai_api_key: Optional[str] = None,
		kky_ollama_uname: Optional[str] = None,
		kky_ollama_password: Optional[str] = None,
		kky_ollama_server: Optional[str] = None,
	) -> None:
		self.openai = openai
		self.model = model
		self.prompt = prompt
		self.command = command
		self.grayscale = grayscale
		self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
		self.kky_ollama_uname = kky_ollama_uname or os.environ.get("KKY_OLLAMA_UNAME")
		self.kky_ollama_password = kky_ollama_password or os.environ.get("KKY_OLLAMA_PASSWORD")
		self.kky_ollama_server = (
			kky_ollama_server
			or os.environ.get("KKY_OLLAMA_SERVER")
			or DEFAULT_KKY_OLLAMA_SERVER
		)

	def predict(self, image_path: str | Path) -> Optional[str]:
		"""Run one zero-shot recognition and return a validated code, or None."""
		image_b64 = encode_image(image_path, grayscale=self.grayscale)
		answer = call_llm_on_fly(
			image=image_b64,
			command=self.command,
			prompt=self.prompt,
			openai=self.openai,
			openai_api_key=self.openai_api_key,
			kky_ollama_uname=self.kky_ollama_uname,
			kky_ollama_password=self.kky_ollama_password,
			kky_ollama_server=self.kky_ollama_server,
			model=self.model,
		)
		code = re.sub(r"\D", "", str(answer.get("seal_number", "")))
		return code if is_plausible_code(code) else None

	def predict_with_voting(self, image_path: str | Path, attempts: int = 3) -> Optional[str]:
		"""Query the model several times and keep the most frequent valid answer."""
		votes = [self.predict(image_path) for _ in range(attempts)]
		valid_votes = [vote for vote in votes if vote is not None]
		if not valid_votes:
			return None
		return Counter(valid_votes).most_common(1)[0][0]
