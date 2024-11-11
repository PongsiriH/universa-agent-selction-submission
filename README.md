<div align="center">
<h1>Agent Selection Challenge</h1>

<h3>Deadline for submission: 11.11.24 23:59 CET</h3>
<h4>Submit codebase via <a href="mailto:challenge@universa.org">official email</a</h4>
</div>

---
This is a submission by Chillax. All of the solutions can be found in `solutions` folder, other than that, we keep most thing the same with only tiny changes for compatabilities. 
- `solutions/selection_algorithm.py` contain Algorithm class compatible with the provided Benchmark class. This contain the main algorithm descripted in the report.
- `solutions/custom_embedder.py` include SentenceTransformerEF, TfIdfEF, and EnsembleEF. These are subclass of BaseEmbedder from Universa.
- `solutions/preprocess_agent_description.py` contain a functions on preprocessing agents description.
- `solutions/run_benchmark.py` simply run benchmark on Algorithm.

# How to use this repo?
```
git clone https://github.com/PongsiriH/universa-agent-selction-submission.git
cd universa-agent-selction-submission
pip install -r requirements.txt

python solutions/run_benchmark.py
```

Sorry, code is a bit messy due to time limitation. 
Any question, you can email huangpongsiri@gmail.com
