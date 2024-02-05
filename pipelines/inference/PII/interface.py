import os
import time
import torch
import tempfile
import numpy as np
import gradio as gr

from models.papers import PhysicallyIntelligentRNN
from utils.data import NAESSEncoder

device = "mps"
ne = NAESSEncoder()
model = PhysicallyIntelligentRNN().to(device)
model.load_state_dict(torch.load("archive/pii_submission.pt", map_location=device))
model.eval()

def process(input, temp=0.9, target_length=128, include_input=True):
    start = time.time()
    with torch.no_grad():  # disable autograd
        start_seq = ne.encode(input.name)
        # make a segment of target_length from the end of the input
        seg = start_seq[-target_length:]
        input = torch.from_numpy(np.array([seg])).int().to(device)
        outputs = []
        for i in range(target_length):
            print(i, target_length, end="\r")
            output, _ = model(input)
            output = output[:, -1, :]
            output = torch.multinomial(torch.softmax(output / temp, dim=1), num_samples=1)
            outputs.append(output)
            input = torch.cat((input, output), dim=1)[:, 1:]

        final_output = torch.cat(outputs, dim=1)
        final_output = final_output[:, -target_length:]
    if include_input:
        final_output = torch.cat((torch.tensor([start_seq]).int().to(device), final_output), dim=1)
    midi_out = ne.decode(final_output.cpu().numpy()[0])
    temp_dir = tempfile.mkdtemp()
    output_midi_path = os.path.join(temp_dir, 'output.mid')
    midi_out.dump(output_midi_path)
    print(f"Completed in {time.time() - start} seconds")
    return output_midi_path



with gr.Blocks(theme=gr.themes.Soft()) as iface:
    gr.Markdown(
    """
    # PII Demo
    """)
    with gr.Row():
        with gr.Column():
            input_midi = gr.File(label="MIDI in")
            with gr.Accordion("Details", open=False):
                temp = gr.Slider(label="temp",  value=0.95, minimum=0.1, maximum=1.0)
                target_length = gr.Slider(label="seq_len", value=512, minimum=32, maximum=2048)
                include_input = gr.Checkbox(label="include start input in output",  value=True)
            process_btn = gr.Button("Process")
        with gr.Column():
            output = gr.File(label="MIDI out")
    process_btn.click(fn=process, inputs=[input_midi, temp, target_length, include_input], outputs=output, api_name="process")
iface.launch()
