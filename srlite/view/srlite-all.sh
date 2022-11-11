#
echo Alaska ilab203
time singularity run -B /explore/nobackup/projects/ilab/data/srlite/products,/explore/nobackup/people/mmacande/srlite/srlite_shared/ccdc_v20221001/Alaska,/explore/nobackup/projects/ilab/projects/CloudMask/products/srlite/v1.2,/explore/nobackup/projects/ilab/data/srlite/toa/Alaska,/explore/nobackup/people/gtamkin/.conda/envs/ilab-gt-dashboard/bin /explore/nobackup/people/iluser/ilab_containers/dev/ilab-base-gdal-srlite-0.9.15.sif python /usr/local/ilab/srlite/srlite/view/SrliteWorkflowCommandLineView.py -toa_dir /explore/nobackup/projects/ilab/data/srlite/toa/Alaska -target_dir /explore/nobackup/people/mmacande/srlite/srlite_shared/ccdc_v20221001/Alaska -cloudmask_dir /explore/nobackup/projects/ilab/projects/CloudMask/products/srlite/v1.2/Alaska -bandpairs "[['blue_ccdc', 'BAND-B'], ['green_ccdc', 'BAND-G'], ['red_ccdc', 'BAND-R'], ['nir_ccdc', 'BAND-N']]" -output_dir /explore/nobackup/projects/ilab/data/srlite/products/srlite-0.9.15-10282022-qa/10282022-all/Alaska --regressor rma --debug 1 --pmask --cloudmask --clean --csv --band8&
#
echo Howland ilab204
time singularity run -B /explore/nobackup/projects/ilab/data/srlite/products,/explore/nobackup/people/mmacande/srlite/srlite_shared/ccdc_v20221001/Howland,/explore/nobackup/projects/ilab/projects/CloudMask/products/srlite/v1.2,/explore/nobackup/projects/ilab/data/srlite/toa/Howland,/explore/nobackup/people/gtamkin/.conda/envs/ilab-gt-dashboard/bin /explore/nobackup/people/iluser/ilab_containers/dev/ilab-base-gdal-srlite-0.9.15.sif python /usr/local/ilab/srlite/srlite/view/SrliteWorkflowCommandLineView.py -toa_dir /explore/nobackup/projects/ilab/data/srlite/toa/Howland -target_dir /explore/nobackup/people/mmacande/srlite/srlite_shared/ccdc_v20221001/Howland -cloudmask_dir /explore/nobackup/projects/ilab/projects/CloudMask/products/srlite/v1.2/Howland -bandpairs "[['blue_ccdc', 'BAND-B'], ['green_ccdc', 'BAND-G'], ['red_ccdc', 'BAND-R'], ['nir_ccdc', 'BAND-N']]" -output_dir /explore/nobackup/projects/ilab/data/srlite/products/srlite-0.9.15-10282022-qa/10282022-all/Howland --regressor rma --debug 1 --pmask --cloudmask --clean --csv --band8&
#
echo Laselva ilab205
time singularity run -B /explore/nobackup/projects/ilab/data/srlite/products,/explore/nobackup/people/mmacande/srlite/srlite_shared/ccdc_v20221001/Laselva,/explore/nobackup/projects/ilab/projects/CloudMask/products/srlite/v1.2,/explore/nobackup/projects/ilab/data/srlite/toa/Laselva,/explore/nobackup/people/gtamkin/.conda/envs/ilab-gt-dashboard/bin /explore/nobackup/people/iluser/ilab_containers/dev/ilab-base-gdal-srlite-0.9.15.sif python /usr/local/ilab/srlite/srlite/view/SrliteWorkflowCommandLineView.py -toa_dir /explore/nobackup/projects/ilab/data/srlite/toa/Laselva -target_dir /explore/nobackup/people/mmacande/srlite/srlite_shared/ccdc_v20221001/Laselva -cloudmask_dir /explore/nobackup/projects/ilab/projects/CloudMask/products/srlite/v1.2/Laselva -bandpairs "[['blue_ccdc', 'BAND-B'], ['green_ccdc', 'BAND-G'], ['red_ccdc', 'BAND-R'], ['nir_ccdc', 'BAND-N']]" -output_dir /explore/nobackup/projects/ilab/data/srlite/products/srlite-0.9.15-10282022-qa/10282022-all/Laselva --regressor rma --debug 1 --pmask --cloudmask --clean --csv --band8&
#
echo RailroadValley ilab206
time singularity run -B /explore/nobackup/projects/ilab/data/srlite/products,/explore/nobackup/people/mmacande/srlite/srlite_shared/ccdc_v20221001/RailroadValley,/explore/nobackup/projects/ilab/projects/CloudMask/products/srlite/v1.2,/explore/nobackup/projects/ilab/data/srlite/toa/RailroadValley,/explore/nobackup/people/gtamkin/.conda/envs/ilab-gt-dashboard/bin /explore/nobackup/people/iluser/ilab_containers/dev/ilab-base-gdal-srlite-0.9.15.sif python /usr/local/ilab/srlite/srlite/view/SrliteWorkflowCommandLineView.py -toa_dir /explore/nobackup/projects/ilab/data/srlite/toa/RailroadValley -target_dir /explore/nobackup/people/mmacande/srlite/srlite_shared/ccdc_v20221001/RailroadValley -cloudmask_dir /explore/nobackup/projects/ilab/projects/CloudMask/products/srlite/v1.2/RailroadValley -bandpairs "[['blue_ccdc', 'BAND-B'], ['green_ccdc', 'BAND-G'], ['red_ccdc', 'BAND-R'], ['nir_ccdc', 'BAND-N']]" -output_dir /explore/nobackup/projects/ilab/data/srlite/products/srlite-0.9.15-10282022-qa/10282022-all/RailroadValley --regressor rma --debug 1 --pmask --cloudmask --clean --csv --band8&
#
echo Senegal ilab207
time singularity run -B /explore/nobackup/projects/ilab/data/srlite/products,/explore/nobackup/people/mmacande/srlite/srlite_shared/ccdc_v20221001/Senegal,/explore/nobackup/projects/ilab/projects/CloudMask/products/srlite/v1.2,/explore/nobackup/projects/ilab/data/srlite/toa/Senegal,/explore/nobackup/people/gtamkin/.conda/envs/ilab-gt-dashboard/bin /explore/nobackup/people/iluser/ilab_containers/dev/ilab-base-gdal-srlite-0.9.15.sif python /usr/local/ilab/srlite/srlite/view/SrliteWorkflowCommandLineView.py -toa_dir /explore/nobackup/projects/ilab/data/srlite/toa/Senegal -target_dir /explore/nobackup/people/mmacande/srlite/srlite_shared/ccdc_v20221001/Senegal -cloudmask_dir /explore/nobackup/projects/ilab/projects/CloudMask/products/srlite/v1.2/Senegal -bandpairs "[['blue_ccdc', 'BAND-B'], ['green_ccdc', 'BAND-G'], ['red_ccdc', 'BAND-R'], ['nir_ccdc', 'BAND-N']]" -output_dir /explore/nobackup/projects/ilab/data/srlite/products/srlite-0.9.15-10282022-qa/10282022-all/Senegal --regressor rma --debug 1 --pmask --cloudmask --clean --csv --band8&
#
echo Serc ilab208
time singularity run -B /explore/nobackup/projects/ilab/data/srlite/products,/explore/nobackup/people/mmacande/srlite/srlite_shared/ccdc_v20221001/Serc,/explore/nobackup/projects/ilab/projects/CloudMask/products/srlite/v1.2,/explore/nobackup/projects/ilab/data/srlite/toa/Serc,/explore/nobackup/people/gtamkin/.conda/envs/ilab-gt-dashboard/bin /explore/nobackup/people/iluser/ilab_containers/dev/ilab-base-gdal-srlite-0.9.15.sif python /usr/local/ilab/srlite/srlite/view/SrliteWorkflowCommandLineView.py -toa_dir /explore/nobackup/projects/ilab/data/srlite/toa/Serc -target_dir /explore/nobackup/people/mmacande/srlite/srlite_shared/ccdc_v20221001/Serc -cloudmask_dir /explore/nobackup/projects/ilab/projects/CloudMask/products/srlite/v1.2/Serc -bandpairs "[['blue_ccdc', 'BAND-B'], ['green_ccdc', 'BAND-G'], ['red_ccdc', 'BAND-R'], ['nir_ccdc', 'BAND-N']]" -output_dir /explore/nobackup/projects/ilab/data/srlite/products/srlite-0.9.15-10282022-qa/10282022-all/Serc --regressor rma --debug 1 --pmask --cloudmask --clean --csv --band8 --cloudmask_suffix toa.cloudmask.tif
#
#Siberia
echo Siberia ilab209
time singularity run -B /explore/nobackup/projects/ilab/data/srlite/products,/explore/nobackup/people/mmacande/srlite/srlite_shared/ccdc_v20221001/Siberia,/explore/nobackup/projects/ilab/projects/CloudMask/products/srlite/v1.2,/explore/nobackup/projects/ilab/data/srlite/toa/Siberia,/explore/nobackup/people/gtamkin/.conda/envs/ilab-gt-dashboard/bin /explore/nobackup/people/iluser/ilab_containers/dev/ilab-base-gdal-srlite-0.9.15.sif python /usr/local/ilab/srlite/srlite/view/SrliteWorkflowCommandLineView.py -toa_dir /explore/nobackup/projects/ilab/data/srlite/toa/Siberia -target_dir /explore/nobackup/people/mmacande/srlite/srlite_shared/ccdc_v20221001/Siberia -cloudmask_dir /explore/nobackup/projects/ilab/projects/CloudMask/products/srlite/v1.2/Siberia -bandpairs "[['blue_ccdc', 'BAND-B'], ['green_ccdc', 'BAND-G'], ['red_ccdc', 'BAND-R'], ['nir_ccdc', 'BAND-N']]" -output_dir /explore/nobackup/projects/ilab/data/srlite/products/srlite-0.9.15-10282022-qa/10282022-all/Siberia --regressor rma --debug 1 --pmask --cloudmask --clean --csv --band8&
#
#Whitesands
echo Whitesands ilab210
time singularity run -B /explore/nobackup/projects/ilab/data/srlite/products,/explore/nobackup/people/mmacande/srlite/srlite_shared/ccdc_v20221001/Whitesands,/explore/nobackup/projects/ilab/projects/CloudMask/products/srlite/v1.2,/explore/nobackup/projects/ilab/data/srlite/toa/Whitesands,/explore/nobackup/people/gtamkin/.conda/envs/ilab-gt-dashboard/bin /explore/nobackup/people/iluser/ilab_containers/dev/ilab-base-gdal-srlite-0.9.15.sif python /usr/local/ilab/srlite/srlite/view/SrliteWorkflowCommandLineView.py -toa_dir /explore/nobackup/projects/ilab/data/srlite/toa/Whitesands -target_dir /explore/nobackup/people/mmacande/srlite/srlite_shared/ccdc_v20221001/Whitesands -cloudmask_dir /explore/nobackup/projects/ilab/projects/CloudMask/products/srlite/v1.2/Whitesands -bandpairs "[['blue_ccdc', 'BAND-B'], ['green_ccdc', 'BAND-G'], ['red_ccdc', 'BAND-R'], ['nir_ccdc', 'BAND-N']]" -output_dir /explore/nobackup/projects/ilab/data/srlite/products/srlite-0.9.15-10282022-qa/10282022-all/Whitesands --regressor rma --debug 1 --pmask --cloudmask --clean --csv --band8&

