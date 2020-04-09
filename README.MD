HOW TO RUN ?

1) For generating the definitions:
   1.1)run : python src\generate_defs_corpus.py 
   It reads config from .\config\config_defs.yaml 
   The source file is '.\data\all_sources_metadata_2020-03-13.csv.'
   The extraction is done from sciScpacy model. The extraction itself is 
   time consuming , hence fr=or each iteration it saves the defintion at 
   temp location and later run the script mentioned in 1.2 to get the final output.
   
   For sciSpcay set-up use this link:
   https://allenai.github.io/scispacy/
   
   1.2) run: python src\create_definition_file.py 
   It reads config from .\config\config_defs.yaml and generates output 
   at .\data\definition-corona-final.csv
   
2) For semantic role tagging using BERT

    run: python src\ser_bert_tagger.py --mode train\eval
    This implementation uses BERT to produce the ser tags. 
    It reads config from ./config/config_defs.yaml .
    Read about BERT here :
    https://github.com/google-research/bert
    
    Download the trained model from here :
    https://drive.google.com/open?id=1p_sb3yxr_n6IBfs2QqlKKIO2Ygk5GBNk 
    
   NOTE : You may need to unzip the files in .\data and place directly under .\data
   folder for processing or you can download the data and model from the link shared above.
   
   This implementation usage tensorflow version 1.15 and python 3.7.
   Download all requirements pip install -r requirements.txt
   

    

    
   