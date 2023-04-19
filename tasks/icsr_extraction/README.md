We support ICSR extraction via decoder-encoder models (`T5` etc.
We will support decoder-only models (`GPT` etc.) soon.

The training arguments are saved in `config.yaml` but can be passed via the command line using `python ../config_to_cmd.py config.yaml`.
Don't forget the backticks as they are crucial to first parse the config file.
        
        python run_summarization_for_icsr_extraction.py `python ../config_to_cmd.py config.yaml`

For prediction, we use a beam search with `num_beams = 5` and a repetition penalty. Additionally, we force the model to adhere to the structure of an ICSR by setting `force_words = ["serious:", "patientsex:", "drugs:", "reactions:"]`. If you change the structure of the ICSR string, be sure to reflect these changes by changing `force_words`.

Evaluate your model predictions on the test set using the custom metric.

        python evaluate_icsr_extraction.py ../../checkpoints/<your-model-folder>/generated_predictions.txt FAERS-PubMed/BioDEX-ICSR