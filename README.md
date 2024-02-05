#  nime gen dnn ml reactivations

### content

`/archive/` contains the pytorch .pt files of the trained models.

`/datasets/` the dir where the datasets are stored when data prep scripts are ran. (We refer to local copies of [Music21 Bach Chorales](https://web.mit.edu/music21/doc/moduleReference/moduleCorpusChorales.html), [Magenta Maestro](https://magenta.tensorflow.org/datasets/maestro), [Magenta Groove](https://magenta.tensorflow.org/datasets/groove), and [Lakh MIDI](https://colinraffel.com/projects/lmd/)). 

`/models/core/` contains common components.

`/models/papers/` contains the paper models.

`/pipelines/inference/` contains the gradio interfaces for each paper.

`/samples/` MIDI output samples.

`/training/` data preparation and training scripts for each paper.

`/utils/` contains data, encoding/decodins, training, inference, and other useful reusable functions.


### setup

`python -m venv venv` sets up a virtual environment.
`source venv/bin/activate` (or equivalent for your OS)

`pip install torch` see pytorch website for the correct version for your OS (+ CUDA version).
`pip install -r requirements.txt` installs the rest of the dependencies.

### use
##### dataprep and training
`python -m training.<paper>.<script>` 

e.g. `python -m training.ScoreTransformer.prepare_data` runs the data prep script for the ScoreTransformer paper.

e.g. `python -m training.ScoreTransformer.train` runs the training script for the ScoreTransformer paper.

##### interface
`python -m pipelines.inference.<paper>.interface` 
 
e.g. `python -m pipelines.inference.RhythmTransformer.interface` runs the interface for the PII paper.

#### use model in pipeline

`pip install -e .` installs the package in editable mode. Then in your own project you can import the model like so:
`from nimegendnn.models.papers import <paper_model_name>` 

```python
from nimegendnn.models.papers import ConvincingHarmony
from nimegendnn.utils.data import NAESSEncoder

device = "mps"

ne = NAESSEncoder()
ne = NAESSEncoder()
model = ConvincingHarmony().to(device)
model.load_state_dict(torch.load("archive/ch_submission.pt", map_location=device))

# ... model specific sampling loop ...
```

### models

| name | ref | paper dir | additional notes | archive/.pt |
| --- | --- | --- | --- | --- |
| BachDuet  | [1] | benatatos_2020 | does not implement Stack Memory | bachduet_submission |
| Convincing Harmony  | [2] | faitas_2019 | | ch_submission |
| WhatHowPlayVAE | [3] | gillick_2021 | | whathowplayauxvae_submission |
| ScoreTransformer | [4] | lupker_2021 | | scoretransformer_submission |
| RhythmTransformerXL | [5] | nuttall_2021 |  | rhythmtransformer_submission |
| LatentDrummer | [6] | warren_2022 | not trained: unclear on data prep | X |
| DrumRBM | [7] | vogl_2017 |  | drbm_submission |
| PII | [8] | naess_2019 | extra for fun | pii_submission |

### references

[1] C. Benetatos, J. VanderStel, and Z. Duan. Bachduet: A deep learning system for human-machine counterpoint improvisation. In R. Michon and F. Schroeder, editors, Proceedings of the International Conference on New Interfaces for Musical Expression, pages 635–640, Birmingham, UK, July 2020. Birmingham City University.

[2] A. Faitas, S. E. Baumann, T. R. Næss, J. Torresen, and C. P. Martin. Generating convincing harmony parts with simple long short-term memory networks. In M. Queiroz and A. X. Sedo ́, editors, Proceedings of the International Conference on New Interfaces for Musical Expression, pages 325–330, Porto Alegre, Brazil, June 2019. UFRGS.

[3] J. Gillick and D. Bamman. What to play and how to play it: Guiding generative music models with multiple demonstrations. In Proceedings of the International Conference on New Interfaces for Musical Expression, Shanghai, China, June 2021.

[4] J. A. T. Lupker. Score-transformer: A deep learning aid for music composition. In Proceedings of the International Conference on New Interfaces for Musical Expression, Shanghai, China, June 2021.

[5] T. Nuttall, B. Haki, and S. Jorda. Transformer neural networks for automated rhythm generation. In Proceedings of the International Conference on New Interfaces for Musical Expression, Shanghai, China, June 2021.

[6] N. Warren and A. C ̧ amci. Latent drummer: A new abstraction for modular sequencers. In Proceedings of the International Conference on New Interfaces for Musical Expression, The University of Auckland, New Zealand, jun 2022.

[7] R. Vogl and P. Knees. An intelligent drum machine for electronic dance music production and performance. In Proceedings of the International Conference on New Interfaces for Musical Expression, pages 251–256, Copenhagen, Denmark, 2017. Aalborg University Copenhagen.

[8] T. R. Næss and C. P. Martin. A physical intelligent instrument using recurrent neural networks. In 19th International Conference on New Interfaces for Musical Expression, NIME 2019, Porto Alegre, Brazil, June 3-6, 2019, pages 79–82. nime.org, 2019.