# Folder 3 — What to show at the exam

Open this page first. The files below are listed in the order you can explain them.

| Order | What I did | Files and what they show |
|---|---|---|
| 1. Prepare the input | I turned the training text into token IDs. | [fineweb_corpus.txt](01_Prepare_the_input/fineweb_corpus.txt) is the text. [fineweb_tokens.txt](01_Prepare_the_input/fineweb_tokens.txt) contains the numeric token IDs. |
| 2. Train the model | I trained a small Transformer on my CPU to predict the next token. | [requirements.txt](02_Train_the_model/requirements.txt) lists the packages. [tiny_transformer.py](02_Train_the_model/tiny_transformer.py) defines the model. [train.py](02_Train_the_model/train.py) runs training. |
| 3. Check training | I checked how the loss and samples changed during training. | [training_log.csv](03_Check_training/training_log.csv) records each update. [loss_plot.png](03_Check_training/loss_plot.png) shows the trend. [generated_samples.txt](03_Check_training/generated_samples.txt) lets you compare text before and during training and at the end. [tiny_transformer.pt](03_Check_training/tiny_transformer.pt) stores the trained weights and vocabulary. |
| 4. Test two prompts | I tried an outside prompt, then an opening from the corpus. | [sample_sky.py](04_Test_two_prompts/sample_sky.py) tried “The sky is”. [sky_prompt_failed.txt](04_Test_two_prompts/sky_prompt_failed.txt) records the missing token. [sample_corpus.py](04_Test_two_prompts/sample_corpus.py) finds an opening the model can process. [corpus_completions.txt](04_Test_two_prompts/corpus_completions.txt) records the opening and 100 completions. |

**What to say in about 30 seconds**

> I trained a small model on tokenized FineWeb text to predict the next token. It ran for about 60 seconds on my CPU and completed 890 updates. Training loss went from 7.71 on the first update to 0.27 on the last. That shows it got better at predicting tokens in the training text, but it doesn't tell us how well it handles new text. “The sky is” failed because one token was missing from the saved vocabulary. The corpus prompt used tokens the model knew, so it generated 100 completions, though some drifted into unrelated phrases.

**If asked to show evidence**, open the loss plot first, then show the failed sky prompt and the first few corpus completions. The training log and checkpoint are there if the examiner asks for details.
