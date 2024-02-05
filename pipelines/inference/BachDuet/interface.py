import os
import time
import torch
import tempfile
import numpy as np
import gradio as gr

from utils.data import BachDuetData, join_midi
from models.papers import BachDuet

device = "mps"
bdd = BachDuetData()
model = BachDuet().to(device)
model.load_state_dict(torch.load("archive/bachduet_submission.pt", map_location=device))
model.eval()

def process(input, temp=0.9):
    start = time.time()
    with torch.no_grad():  # disable autograd
        start_seq = bdd.encode(input.name) #(4704, 3)
        rhythm = start_seq[:, 0]
        cpc = start_seq[:, 1]
        midi = start_seq[:, 2]
        # make batch of 1
        target_length = len(start_seq) # 4704
        # make a segment of target_length from the end of the input
        seg = start_seq #[-target_length:]
        token_preds = []
        key_preds = []
        seq_selection_len = 32
        for i in range(target_length):
            print(i, "/", target_length)
            rhythm_in = torch.from_numpy(np.array([rhythm[:seq_selection_len]])).long().to(device)
            cpc_in = torch.from_numpy(np.array([cpc[:seq_selection_len]])).long().to(device)
            midi_in = torch.from_numpy(np.array([midi[:seq_selection_len]])).long().to(device)
            # print("rhythm", rhythm_in.shape, "cpc", cpc_in.shape, "midi", midi_in.shape)

            token_pred, token_pred_sm, key_pred, key_pred_sm = model(rhythm_in, cpc_in, midi_in)
            token_pred = token_pred[:, -1, :] # get last token
            token_pred = torch.multinomial(torch.softmax(token_pred / temp, dim=1), num_samples=1)
            token_preds.append(token_pred)
            key_pred = key_pred[:, -1, :] # get last token
            key_pred = torch.multinomial(torch.softmax(key_pred / temp, dim=1), num_samples=1)
            key_preds.append(key_pred)
            
            # dont update rhythm as we are not using it
            # add key to cpc and pop first element
            cpc = np.append(cpc, key_pred.cpu().numpy())
            cpc = np.delete(cpc, 0)
            # add token to midi and pop first element
            midi = np.append(midi, token_pred.cpu().numpy())
            midi = np.delete(midi, 0)
            
        token_preds = torch.cat(token_preds, dim=1) 
        key_preds = torch.cat(key_preds, dim=1)
    rhythm = start_seq[:target_length, 0]
    decode_data = np.stack([rhythm, key_preds.cpu().numpy()[0], token_preds.cpu().numpy()[0]], axis=1)
    print("decode_data", decode_data.shape) #(4704, 3)
    midi_out = bdd.decode(decode_data)
    temp_dir = tempfile.mkdtemp()
    output_midi_path = os.path.join(temp_dir, 'bach_duet_out.mid')
    midi_out.dump(output_midi_path)
    # joined_midi = join_midi([input.name, output_midi_path])
    # joined_midi_path = os.path.join(temp_dir, 'bach_duet_joined.mid')
    # joined_midi.dump(joined_midi_path)
    # print(f"Completed in {time.time() - start} seconds")
    return output_midi_path


with gr.Blocks(theme=gr.themes.Soft()) as iface:
    gr.Markdown(
    """
    # BachDuet Demo
    """)
    with gr.Row():
        with gr.Column():
            input_midi = gr.File(label="MIDI in (soprano)")
            with gr.Accordion("Details", open=False):
                temp = gr.Slider(label="temp",  value=0.95, minimum=0.1, maximum=1.0)
            process_btn = gr.Button("Process")
        with gr.Column():
            output = gr.File(label="MIDI out (bass)")
    process_btn.click(fn=process, inputs=[input_midi], outputs=output, api_name="process")
iface.launch()
