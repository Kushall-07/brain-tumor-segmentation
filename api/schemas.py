from pydantic import BaseModel


class PredictRequest(BaseModel):
    data_dir: str
    checkpoint_path: str
    output_dir: str
    case_index: int = 0
    save_probabilities: bool = False


class PredictResponse(BaseModel):
    status: str
    result: dict


class PredictUploadResponse(BaseModel):
    status: str
    result: dict
