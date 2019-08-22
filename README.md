# FACILE (FAst CalorImeter LEarning)

## Description
This repository was created by Jack Dinsmore and Jeff Krupa at MIT in the summer of 2019 in order to experiment possible machine learning approaches to Hydrogen Calorimeter rec hit regression. We investigated several dense neural networks, trained on simulated data, and found one that was small enough to be run at reasonable times yet large enough to predict the generator energy of simulated pions with reasonable accuracy.

## Results
The neural network that we finally settled on is named `Model4Exp` in _train-models.py_. It is a four-layer dense neural network which contains 2,111 parameters and takes 36 inputs: eta, phi, depth, gain, and the 8 timeslices of charge, pedestal, NoiseADC, and NoisePhoto. With a batch size of 10,000 images, it can run predictions on MIT's Tier 3 computing cluster at about 500 ns per event on a CPU.

## Use
The code in this repository is easily adaptable if you wish to test your own dense neural network. Simply follow these steps.
1. Run `cmsenv` if you are on MIT's Tier 3 system so that you will have access to `ROOT`
1. **Gather data** The model must have a train dataset. You can find one at `/data/t3home000/jkrupa/rh_studies/Out.root_skimmedRH2.root` on MIT's Tier 3 system, or you can create your own with a similar format. Once you have saved your data as a `.root` file, run `python2 load-root.py --filename <<DIR>> --outdir output/` where `<<DIR>>` is the location of your `.root` training dataset. If you wish to use inputs other than those listed in the "Results" section, modify the list called `inputs` in `load-root.py` before you run the above command so that `inputs` contains only the inputs you wish your model to take.
2. **Create your model** Create your model in the same format as all the other models defined at the top of `train-models.py`. Create a new subclass of `mc.ClassModel` and in it create a function called `get_outputs(self)` which creates a keras Model and returns it. Make sure to set `self.name` to something unique: this will be the name of the file in which your model data is stored. Then add the name of your model to the list named `MODELS`, below all the model definitions in `train-models.py`. Finally,  run `python2 train-models.py --train --plot`.
3. **Getting the performance of your model** To generate the response, response-corrected-resolution, and timing plots for your new model, execute `python2 performance-models.py --pickle models/evt/v0 --figdir <<FIGDIR>>` where `<<FIGDIR>>` is the place where you wish to write all the files describing your data and the results of your model. In `<<FIGDIR>>`, you will find many images of various PU, pt bins of the train data, several images called *resolution_x.png* and *response_x.png* which contain the resolution and response curves respectively, and *timing.png*, which contains speed of prediction for the model per image as a function of batch size. For high batch size, this *timing.png* plot tends to become inaccurate; to resolve this, repeat step 3 with the added flag `--trials <<N>>` where `<<N>>` is the number of times you wish to run predictions. I usually do 10. You may omit the `--train` flag in this case because the model will already have been trained.

`64657374726f79206d616869`
