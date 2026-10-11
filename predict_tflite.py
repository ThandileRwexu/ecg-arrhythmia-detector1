import numpy as np
import tensorflow as tf

class TFLitePredictor:
    def __init__(self, model_path='ecg_cnn_focal_int8.tflite'):  # ← Change here
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.class_names = ['Normal (N)', 'Supraventricular (SVEB)', 'Ventricular (VEB)']

        # Check if model is quantized (INT8 input)
        self.is_quantized = self.input_details[0]['dtype'] == np.int8

        # Get quantization parameters if quantized
        if self.is_quantized:
            self.input_scale, self.input_zero_point = self.input_details[0]['quantization']
            self.output_scale, self.output_zero_point = self.output_details[0]['quantization']

    def predict(self, beat):
        # Normalize beat (z-score)
        beat = np.array(beat, dtype=np.float32).flatten()
        beat = (beat - np.mean(beat)) / (np.std(beat) + 1e-8)
        beat = beat.reshape(1, 180, 1).astype(np.float32)

        # Quantize input if needed (FLOAT32 -> INT8)
        if self.is_quantized:
            beat = (beat / self.input_scale + self.input_zero_point).astype(np.int8)

        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], beat)
        self.interpreter.invoke()

        # Get output
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        # Dequantize output if needed (INT8 -> FLOAT32)
        if self.is_quantized:
            output = (output.astype(np.float32) - self.output_zero_point) * self.output_scale

        # Raw probabilities — NO CLIPPING
            probs = output.astype(np.float32)
            probs = probs / (probs.sum() + 1e-8)  # ensure sum = 1


        # Get prediction
        predicted_class = int(np.argmax(output))
        confidence = float(output[predicted_class])

         # ===== UNCERTAINTY HANDLING =====
        if confidence < 0.60:
         return "Uncertain — needs review", confidence, probs.tolist()

        return self.class_names[predicted_class], confidence, output.tolist()