# MA-PEP
MA-PEP：a novel anticancer peptide prediction framework with multi-modal feature fusion based on attention mechanisms

## How to Use
Add your training and testing dataset in the data directory or use pre-prepared datasets.

Open the `main.py` file in the train folder and modify it according to your data and requirements.

Run the `main.py`file to train the MA-PEP model:

```bash
python train/main.py
```

See `requirements.txt` for details of dependent packages.

## How to Classify a Single Peptide Sequence
If you want to classify a single peptide sequence, you can use the `instance.py`

Create an example input for the peptide sequence and pass it to the model for classification

```python
# Example peptide sequence
peptide_sequence = "FAKKLLAKALKL"

```
Run the `instance.py` file to classify the example peptide sequence.
Please note that this is just a simple example, and you will need to customize it to meet your project requirements.

You can also replace the pre-trained MA-PEP model
```python
# load model
state_dict = torch.load('ME-PEP_model.pt')

```

