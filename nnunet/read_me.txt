step 0: Download code of nnunet from "https://github.com/kvpratama/nnunet#examples"
        or "git clone https://github.com/MIC-DKFZ/nnUNet.git
	    cd nnUNet
	    pip install -e ."

step 1: set up the environments

export nnUNet_raw_data_base="/media/fabian/nnUNet_raw"
export nnUNet_preprocessed="/media/fabian/nnUNet_preprocessed"
export RESULTS_FOLDER="/media/fabian/nnUNet_trained_models"

step 2: prepared date for nnunet: run the "prapare_data_for_nnunet.py"

step 3: run the "Task666_endoscopy.py" transfer data for nnunet pre-process


step 4: >nnUNet_plan_and_preprocess -t 666 -pl3d None

step 5: >nnUNet_train 2d nnUNetTrainerV2 666 FOLD  
          (where fold is again 0, 1, 2, 3 and 4 - 5-fold cross validation)