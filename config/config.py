from pydantic import BaseModel, Field
import yaml


class LLMConfig(BaseModel):
    model: str = Field(description="The model to use")
    model_provider: str = Field(description="The model provider to use")
    temperature: float = Field(description="The temperature to use")


class SPE_SCConfig(BaseModel):
    N: int = Field(description="The number of self-correction sequences to run")
    M: int = Field(description="The number of self-correction sequences to run")
    llm_rewriter: LLMConfig = Field(description="The rewriter LLM config")
    llm_filter: LLMConfig = Field(description="The filter LLM config")
    llm_reconstructor: LLMConfig = Field(description="The reconstructor LLM config")


class Config(BaseModel):
    llm_target: LLMConfig = Field(description="The target LLM config")
    spe_sc_config: SPE_SCConfig = Field(description="The SPE_SC config")

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r") as f:
            yaml_data = yaml.safe_load(f)
            return cls.model_validate(yaml_data)