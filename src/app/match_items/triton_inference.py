import asyncio
import os
import numpy as np
import grpc
import tritonclient.grpc.aio as grpcclient
from tritonclient.grpc import service_pb2, service_pb2_grpc
from tritonclient.utils import triton_to_np_dtype
from transformers import BertTokenizer

class InferenceModule:
    """
    Module for establish triton connection
    """
    def __init__(self) -> None:
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = 128
        self.url = os.environ.get("TRITON_SERVER_URL", "localhost:7001")
        self.triton_client = grpcclient.InferenceServerClient(url=self.url)
        self.device = 'cpu'

    async def infer_text(  
        self,
        text1: str,
        text2: str,
        model_name: str = "bert",
    ) -> dict:
        """
        Perform inference on the input texts using model.
        """
        model_meta, model_config = self.parse_model_metadata(model_name)
        
        input_ids_meta = next(inp for inp in model_meta.inputs if inp.name == "input_ids")
        attention_mask_meta = next(inp for inp in model_meta.inputs if inp.name == "attention_mask")
        token_type_ids_meta = next(inp for inp in model_meta.inputs if inp.name == "token_type_ids")
        
        max_length = input_ids_meta.shape[1]
        
        tokenized_data = self.preprocess_text(text1, text2)
        
        inputs = [
            grpcclient.InferInput(
                input_ids_meta.name, 
                [1, max_length], 
                input_ids_meta.datatype
            ),
            grpcclient.InferInput(
                attention_mask_meta.name, 
                [1, max_length], 
                attention_mask_meta.datatype
            ),
            grpcclient.InferInput(
                token_type_ids_meta.name, 
                [1, max_length], 
                token_type_ids_meta.datatype
            )
        ]
        
        inputs[0].set_data_from_numpy(
            tokenized_data['input_ids'].astype(triton_to_np_dtype(input_ids_meta.datatype))
        )
        inputs[1].set_data_from_numpy(
            tokenized_data['attention_mask'].astype(triton_to_np_dtype(attention_mask_meta.datatype))
        )
        inputs[2].set_data_from_numpy(
            tokenized_data['token_type_ids'].astype(triton_to_np_dtype(token_type_ids_meta.datatype))
        )
        
        outputs = [grpcclient.InferRequestedOutput(model_meta.outputs[0].name)]
        
        results = await self.triton_client.infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs,
        )
        
        output = results.as_numpy(model_meta.outputs[0].name)[0]
        
        cls_idx = np.argmax(output)
        cls_logit = output[cls_idx]
        
        probabilities = np.exp(output) / np.sum(np.exp(output))
        
        return {
            "class_id": int(cls_idx), 
            "logit": float(cls_logit),
            "probability": float(probabilities[cls_idx]),
            "all_probabilities": probabilities.tolist()
        }

    def parse_model_metadata(self, model_name: str) -> object:
        """Parse metadata and configuration of the model."""
        channel = grpc.insecure_channel(self.url)
        grpc_stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)
        metadata_request = service_pb2.ModelMetadataRequest(name=model_name)
        metadata_response = grpc_stub.ModelMetadata(metadata_request)

        config_request = service_pb2.ModelConfigRequest(name=model_name)
        config_response = grpc_stub.ModelConfig(config_request)

        return metadata_response, config_response

    def preprocess_text(self, text1: str, text2: str) -> dict:
        """Preprocess text inputs for BERT."""
        encoded = self.tokenizer.encode_plus(
            text1,
            text2,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='np'
        )
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'], 
            'token_type_ids': encoded['token_type_ids']  
        }
