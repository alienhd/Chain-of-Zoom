import argparse
import os
from pathlib import Path
import torch
from PIL import Image # Add if image preprocessing is done here, not strictly needed for text_enc calibration
# Assuming SD3Euler and other necessary components are importable from project structure
from osediff_sd3 import SD3Euler

from optimum.onnxruntime import ORTQuantizer # ORTModelForFeatureExtraction is for inference
from optimum.onnxruntime.configuration import QuantizationConfig #, AutoCalibrationConfig, AutoQuantizationConfig # Removed unused Auto configs for now

# --- Main Function ---
def main():
    parser = argparse.ArgumentParser("Model Quantization Script")
    parser.add_argument("--model_component", type=str, default="text_enc_1", choices=["text_enc_1", "text_enc_2", "text_enc_3", "transformer", "vae_encoder", "vae_decoder"])
    parser.add_argument("--output_dir", type=str, default="ckpt/quantized")
    parser.add_argument("--calibration_data_dir", type=str, default="samples") # For image-derived prompts
    parser.add_argument("--num_calibration_samples", type=int, default=10)
    parser.add_argument('--pretrained_model_name_or_path', type=str, default='stabilityai/stable-diffusion-3-medium-diffusers')
    parser.add_argument('--device_for_quant_script', type=str, default='cpu', help='Device to run this quantization script on, and for loading initial model. ONNX export often prefers CPU.')

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # Base directory for this component's quantization artifacts
    component_quant_base_dir = Path(args.output_dir) / args.model_component
    component_quant_base_dir.mkdir(parents=True, exist_ok=True)

    # Path for the final quantized ONNX model (directory, as Optimum saves multiple files)
    final_quantized_model_save_path = component_quant_base_dir / f"{args.model_component}_quantized"
    final_quantized_model_save_path.mkdir(parents=True, exist_ok=True)

    # Path for the intermediate base ONNX model
    onnx_model_export_path = component_quant_base_dir / f"{args.model_component}_base.onnx"


    # 1. Load PyTorch Model (specifically the component)
    print(f"Loading PyTorch model components from {args.pretrained_model_name_or_path} on device {args.device_for_quant_script}...")
    sd3_model_for_component_extraction = SD3Euler(model_key=args.pretrained_model_name_or_path, device=args.device_for_quant_script)

    pytorch_component = None
    tokenizer_for_component = None
    hf_model_name_for_component = None

    if args.model_component == "text_enc_1":
        pytorch_component = sd3_model_for_component_extraction.text_enc_1
        tokenizer_for_component = sd3_model_for_component_extraction.tokenizer_1
        if hasattr(pytorch_component, 'config') and hasattr(pytorch_component.config, '_name_or_path'):
            hf_model_name_for_component = pytorch_component.config._name_or_path
    elif args.model_component == "text_enc_2":
        pytorch_component = sd3_model_for_component_extraction.text_enc_2
        tokenizer_for_component = sd3_model_for_component_extraction.tokenizer_2
        if hasattr(pytorch_component, 'config') and hasattr(pytorch_component.config, '_name_or_path'):
            hf_model_name_for_component = pytorch_component.config._name_or_path
    elif args.model_component == "text_enc_3": # T5EncoderModel
        pytorch_component = sd3_model_for_component_extraction.text_enc_3
        tokenizer_for_component = sd3_model_for_component_extraction.tokenizer_3
        if hasattr(pytorch_component, 'config') and hasattr(pytorch_component.config, '_name_or_path'):
            hf_model_name_for_component = pytorch_component.config._name_or_path
        elif hasattr(pytorch_component, 'config') and hasattr(pytorch_component.config, 'name_or_path'): # Fallback for T5
             hf_model_name_for_component = pytorch_component.config.name_or_path

        if hf_model_name_for_component:
            print(f"Attempting to use HF name for T5 ({args.model_component}): {hf_model_name_for_component}")
        else:
            # Fallback for T5 if name_or_path is not found directly in expected places
            # This is a known configuration for SD3 medium's T5 component.
            # Note: This is a specific workaround. A more general solution would require robust
            # HF model name detection or manual ONNX export for such components.
            hf_model_name_for_component = "google/t5-v1_1-xl"
            print(f"HF model name for {args.model_component} not found in config, using default: {hf_model_name_for_component}")
            # Important: ORTQuantizer.from_pretrained might expect a model that can perform the "feature-extraction" task.
            # For T5, "text2text-generation" is more common. If "default" or "feature-extraction" fails,
            # this component might need special handling (e.g. manual ONNX export of just the encoder).
    elif args.model_component == "transformer":
        print(f"Quantization for model component '{args.model_component}' is complex and not yet implemented in this script.")
        print("This would likely require manual ONNX export and a custom calibration data pipeline.")
        pytorch_component = None # Explicitly set to None or skip further processing
        hf_model_name_for_component = None
    elif args.model_component == "vae_encoder":
        print(f"Quantization for model component '{args.model_component}' is not yet implemented.")
        pytorch_component = None
        hf_model_name_for_component = None
    elif args.model_component == "vae_decoder":
        print(f"Quantization for model component '{args.model_component}' is not yet implemented.")
        pytorch_component = None
        hf_model_name_for_component = None


    if pytorch_component is None:
        print(f"Could not load PyTorch component: {args.model_component}")
        return
    if hf_model_name_for_component is None:
        print(f"Could not determine Hugging Face name_or_path for component: {args.model_component}. This is needed by ORTQuantizer.from_pretrained.")
        # As a fallback, we might try to export the component to ONNX manually, then use ORTQuantizer.from_pretrained(onnx_path)
        # However, this subtask focuses on the HF name path first.
        return


    pytorch_component.eval()
    print(f"Successfully loaded PyTorch component: {args.model_component} from {hf_model_name_for_component if hf_model_name_for_component else 'loaded instance'}")

    # 2. Prepare Calibration Data
    print("Preparing calibration data...")
    sample_prompts = [
        "a photo of a cat", "a painting of a landscape", "an astronaut riding a horse",
        "detailed portrait of an old man", "futuristic cityscape at night", "a delicious plate of pasta",
        "abstract art with vibrant colors", "a quiet library scene", "a dragon breathing fire",
        "a robot working in a factory"
    ]
    sample_prompts = sample_prompts[:args.num_calibration_samples]

    def calibration_dataloader_for_tokenizer(tokenizer, prompts, batch_size=1):
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            inputs = tokenizer(
                batch_prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
            )
            yield {"input_ids": inputs.input_ids.to(args.device_for_quant_script)}

    # 3. Quantization Configuration
    quantization_config_optimum = QuantizationConfig(
        is_static=True,
        format="QDQ",
        # Other parameters like mode, per_channel can be added if needed
    )

    # 4. Perform Quantization using ORTQuantizer
    try:
        print(f"Initializing ORTQuantizer for {hf_model_name_for_component}...")
        # ORTQuantizer.from_pretrained can take a model name or a path to a PyTorch model instance (if not HF hub)
        # For subcomponents, it's often easier to export to ONNX first, then load that ONNX for quantization.
        # However, the instruction implies using hf_model_name_for_component.

        # Let's try the direct HF model name approach first as per the draft.
        # This assumes 'hf_model_name_for_component' is a valid model on the Hub that Optimum can process.
        # For CLIPTextModel, feature="text-generation" or "feature-extraction" might be suitable.
        # Using "default" or letting Optimum infer.
        quantizer = ORTQuantizer.from_pretrained(hf_model_name_for_component, feature="default")

        print("Creating calibration data generator...")
        calibration_data_generator = calibration_dataloader_for_tokenizer(tokenizer_for_component, sample_prompts)

        # Define calibration configuration for Optimum's quantize method
        # AutoCalibrationConfig can be used if we point it to a dataset
        # For direct generator, we pass it to quantize method's calibration_data
        # calibration_config = AutoCalibrationConfig.create(calibration_data_generator) # This seems not directly used by quantizer.quantize

        print(f"Starting quantization process for {args.model_component}.")
        print(f"Base ONNX model will be exported to: {onnx_model_export_path}")
        print(f"Final quantized model will be saved in: {final_quantized_model_save_path}")

        # The quantize method of ORTQuantizer handles ONNX export if model path is not ONNX
        # and then applies quantization.
        quantizer.quantize(
            save_dir=final_quantized_model_save_path, # This should be the final dir for the quantized model
            quantization_config=quantization_config_optimum,
            # onnx_model_path=onnx_model_export_path, # If we wanted to control intermediate ONNX path
            calibration_data=calibration_data_generator, # Pass the generator directly
        )

        print(f"Quantized model for {args.model_component} saved to {final_quantized_model_save_path}")

    except Exception as e:
        print(f"Error during quantization of {args.model_component} with Optimum: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
