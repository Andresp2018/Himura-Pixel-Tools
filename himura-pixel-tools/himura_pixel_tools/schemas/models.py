"""Model-related schemas (model index, validation, install)."""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field

ModelType = Literal[
    "base", "vae", "lora", "controlnet", "ip_adapter",
    "motion", "segmentation", "upscale",
]
BaseCompat = Literal["sdxl", "sd15", "flux_optional_future", "ideogram_gguf"]
Precision = Literal["fp16", "bf16", "fp32", "int8_optional", "q8_gguf", "q6_gguf", "q5_gguf", "q4_gguf"]


class ModelIndex(BaseModel):
    """One entry in the local model index (spec.local_model_loader_spec.model_index_schema)."""

    model_id: str
    display_name: str
    type: ModelType
    source_url: Optional[str] = None
    local_path: str
    sha256: Optional[str] = None
    license: Optional[str] = None
    base_compatibility: list[BaseCompat] = Field(default_factory=list)
    precision: Precision = "fp16"
    enabled: bool = True
    downloaded_at: Optional[str] = None
    trigger: str = ""
    notes: str = ""


class ValidationReport(BaseModel):
    model_id: str
    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    files_present: list[str] = Field(default_factory=list)
    files_missing: list[str] = Field(default_factory=list)


class VRAMReport(BaseModel):
    device: str
    device_name: str
    total_vram_gb: Optional[float] = None
    allocated_gb: Optional[float] = None
    reserved_gb: Optional[float] = None
    free_gb: Optional[float] = None
    cuda_available: bool
    torch_version: Optional[str] = None
    torch_cuda_version: Optional[str] = None
    diagnostics: list[str] = Field(default_factory=list)
    loaded_model: Optional[str] = None
    active_adapters: list[str] = Field(default_factory=list)


class InstallModelRequest(BaseModel):
    """Download or import a model into the local store."""

    source_url: Optional[str] = None        # HF repo id or https URL
    model_type: ModelType = "base"
    display_name: Optional[str] = None
    local_path: Optional[str] = None        # for import of an existing file
    license: Optional[str] = None
    precision: Precision = "fp16"
    base_compatibility: list[BaseCompat] = Field(default_factory=lambda: ["sdxl"])
    revision: Optional[str] = None
    variant: Optional[str] = None


