# V-Net running script, implemented with keras


Install the necessary packages as follows:<br>
```pip3 install -r requirements.txt```

Key features:<br>
- Implemented a modified V-Net<br>
- Use on-the-fly data augmentation (translate, zoom, shear, flip, and rotate)<br>
- Uses group normalization (default group size 8)<br>

Help:
```bash
usage: run_vnet3d.py [-h] --core_tag CORE_TAG --nii_dir NII_DIR --batch_size
                     BATCH_SIZE --image_size IMAGE_SIZE --learning_rate
                     LEARNING_RATE --group_size GROUP_SIZE --f_root F_ROOT
                     --n_validation N_VALIDATION --n_test N_TEST --optimizer
                     OPTIMIZER [--print_summary_only]

Script to run UNet3D

optional arguments:
  -h, --help            show this help message and exit
  --core_tag CORE_TAG, -ct CORE_TAG
  --nii_dir NII_DIR, -I NII_DIR
  --batch_size BATCH_SIZE, -bs BATCH_SIZE
  --image_size IMAGE_SIZE, -is IMAGE_SIZE
  --learning_rate LEARNING_RATE, -lr LEARNING_RATE
  --group_size GROUP_SIZE, -gs GROUP_SIZE
  --f_root F_ROOT, -fr F_ROOT
  --n_validation N_VALIDATION
  --n_test N_TEST
  --optimizer OPTIMIZER, -op OPTIMIZER
  --print_summary_only
```

