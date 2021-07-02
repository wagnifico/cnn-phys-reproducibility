# Acoustic propagation using deep learning methods

## Install dependencies

We recommend using a `conda environment` to install all the project dependencies, listed in the [requirements.txt](requirements.txt) file.
Module `mpi4py` is only used in inference, so it is not presented in the requirements list.

**Disclaimer**: This project has been tested on Ubuntu 18.04 and in a Red Hat Enterprise Linux Server release 7.5 (Maipo).

<pre style="background-color:#cff0fa"><code>conda create --name &#60;myEnvName&#62;
conda activate &#60;myEnvName&#62;
conda install pip
pip install -r requirements.txt
</code></pre>

## Generating the data

You can generate the databases of acoustic propagating waves with the Palabos library project available [here](../simulation/). It is also possible to use the network with any set of 2D fields as long as they are stored as `.vti` files and the corresponding name of channels are set in configuration files.

## Preprocessing the data

First you need to preprocess the previously generated `.vti` files into `.pt` (native PyTorch format), so that the loading of data is done more efficiently. The `configuration_train.yaml` file has the following parameters that affect the data preprocessing step:

**Data parameters :**
This category of params handles the location of the data, as well as the format of files.

<table border="1" align="center">
  <tbody>
    <tr style="background-color:#82afbd">
      <td align="center"><b>parameter</b></td>
      <td align="center"><b>example</b></td>
      <td align="center"><b>description</b></td>
    </tr>
    <tr>
      <td><code>dataset</code></td>
      <td align="center"><code>'mydatabase'</code></td>
      <td>dataset name, to be used on files and folders</td>
    </tr>
    <tr>
      <td><code>dataPathRaw</code></td>
      <td align="center"><code>'path/to/raw/database'</code></td>
      <td>path to the folder with the <b>raw</b> database (simulation data) for training</td>
    </tr>
    <tr>
      <td><code>dataPath</code></td>
      <td align="center"><code>'path/to/datbase'</code></td>
      <td>path to the folder with <b>preprocessed</b> database for training</td>
    </tr>
    <tr>
      <td><code>formatVTK</code></td>
      <td align="center"><code>'{:06d}/vtk{:06d}.vt'</code></td>
      <td>format of the raw (simulation) data filename</code></td>
    </tr>
    <tr>
      <td><code>formatPT</code></td>
      <td align="center"><code>'scene{:06d}_group{:05d}.pt'</code></td>
      <td>format of the preprocessed data filename</td>
    </tr>
    <tr>
      <td><code>frameStep</code></td>
      <td align="center"><code>1</code></td>
      <td>interval between solution steps (must be a multiple of the LBM recorded values)</td>
    </tr>
    </tr>
    <tr>
      <td><code>numWorkers</code></td>
      <td align="center"><code>0</code></td>
      <td>number of parallel workers for dataloader (if 0, selected by pytorch)</td>
    </tr>
  </tbody>
</table>

**Model parameters :**
Parameters that affect the model definition (e.g. input channels, type of data)

<table border="1" align="center">
  <tbody>
    <tr style="background-color:#82afbd">
      <td align="center"><b>parameter</b></td>
      <td align="center"><b>example</b></td>
      <td align="center"><b>description</b></td>
    </tr>
    <tr>
      <td><code>numInputFrames</code></td>
      <td align="center"><code>1</code></td>
      <td>how many data frames will be used as input</code></td>
    </tr>
    <tr>
      <td><code>numOutputFrames</code></td>
      <td align="center"><code>1</code></td>
      <td>how many data frames will be output (and compared with a target frame during training)</td>
    </tr>
    </tr>
    <tr>
      <td><code>channels</code></td>
      <td align="center"><code>['density', 'velocity-x']</code></td>
      <td>name of the quantities to consider from preprocessing to training. Options: <code>density</code>, <code>velocity-x</code>, <code>velocity-y</code></td>
    </tr>
  </tbody>
</table>

Once configured, you can launch the data preprocessing by executing:

<pre style="background-color:#cff0fa"><code>python train.py --trainingConf configuration.yaml --preproc
</code></pre>

## Training the model

To train the model, edit again the `configuration_train.yaml` file with the parameters listed below. Then run this command:

<pre style="background-color:#cff0fa"><code>python train.py --trainingConf configuration_train.yaml
</code></pre>

Additional flags:

- `--double`: use double (64-bits) precision. Must also be used on the preprocessing
- `--verbose`: display evolution of training and epoch statistics on terminal

**Paramaters**

<table border="1" align="center">
  <tbody>
    <tr style="background-color:#82afbd">
      <td align="center"><b>parameter</b></td>
      <td align="center"><b>example</b></td>
      <td align="center"><b>description</b></td>
    </tr>
    <tr>
      <th style="background-color:#a6e4f7" colspan="3" align="center">Data and output paramaters</th>
    </tr>
    <tr>
      <td><code>numWorkers</code></td>
      <td align="center"><code>0</code></td>
      <td>number of parallel workers for dataloader (if 0, selected by pytorch)</td>
    </tr>
    <tr>
      <td><code>modelDir</code></td>
      <td align="center"><code>'/path/to/model/dir'</code></td>
      <td>output folder for trained model and loss log</td>
    </tr>
    <tr>
      <td><code>version</code></td>
      <td align="center"><code>'test'</code></td>
      <td>experiment version</td>
    </tr>
    <tr>
      <td><code>modelFilename</code></td>
      <td align="center"><code>'convModel'</code></td>
      <td>trained model name</td>
    </tr>
    <tr>
      <th style="background-color:#a6e4f7" colspan="3" align="center">Training parameters</th>
    </tr>
    <tr>
      <td><code>resume</code></td>
      <td align="center"><code>false</code></td>
      <td>Resume training from checkpoint</td>
    </tr>
    <tr>
      <td><code>maxEpochs</code></td>
      <td align="center"><code>100</code></td>
      <td>Maximum number of training epochs</td>
    </tr>
    <tr>
      <td><code>batchSize</code></td>
      <td align="center"><code>32</code></td>
      <td>Batch size</td>
    </tr>
    <tr>
      <td><code>maxDatapoints</code></td>
      <td align="center"><code>0</code></td>
      <td>Maximum number of datapoints (if 0, all available datapoints are used). A datapoint is the group of frames that is composed by the merge of the input and target frames. When the maximun number of datapoints is smaller than the number of available groups, a reduced number of groups per scene (simulation) is selected rather than a few complete scenes, aiming at a more diverse database</td>
    </tr>
    <tr>
      <td><code>selectMode</code></td>
      <td align="center"><code>first</code></td>
      <td>How the group of frames are selected from the scenes during preprocessing. For <strong>n</strong> groups:<br/>
      <ul>
        <li><code>first</code>: first <strong>n</strong> groups (start of the scene)</li>
        <li><code>last</code>: last <strong>n</strong> groups (end of the scene)</li>
        <li><code>random</code>: random <strong>n</strong> groups, uniform distribution</li>
      </ul>
      <code>first</code> and <code>last</code> options will only change behaviour of the preprocessing when the selected number of datapoints is smaller then the number of available groups; <code>random</code> will only change the order of the groups.
      </td>
    </tr>
    <tr>
      <td><code>shuffleTraining</code></td><td align="center"><code>true</code></td><td>Shuffle data batches during training</td>
    </tr>
    <tr>
      <td><code>flipData</code></td><td align="center"><code>true</code></td><td>Flips data fields during training</td>
    </tr>
    <tr>
      <td style="background-color:#a6e4f7" colspan="3" align="center"><b>Optimizer and scheduler parameters</b></td>
    </tr>
    <tr>
      <td style="background-color:#cff0fa" colspan="3" align="center"><ins>Optimizer parameters</ins> - Adam</td>
    </tr>
    <tr>
      <td><code>overrideatresume</code></td>
      <td align="center"><code>false</code></td>
      <td>override optimizer when resuming training</td>
    </tr>
    <tr>
      <td><code>lr</code></td>
      <td align="center"><code>1.0e-4</code></td>
      <td>learning rate</td>
    </tr>
    <tr>
      <td><code>weight_decay</code></td>
      <td align="center"><code>0.0</code></td>
      <td>weight decay</td>
    </tr>
    <tr>
      <td style="background-color:#cff0fa" colspan="3" align="center"><ins>Scheduler parameters</ins></td>
    </tr>
    <tr>
      <td><code>type</code></td>
      <td align="center"><code>'reduceLROnPlateau'</code></td>
      <td>scheduler type</td>
    </tr>
    <tr>
      <td><code>overrideatresume</code></td>
      <td align="center"><code>false</code></td>
      <td>override scheduler when resuming training</td>
    </tr>
    <tr>
      <td style="background-color:#a6e4f7" colspan="3" align="center"><b>Model parameters</b></td>
    </tr>
    <tr>
      <td style="background-color:#cff0fa" colspan="3" align="center"><ins>Data preparation</ins></td>
    <tr>
      <td><code>scalarAdd</code></td>
      <td align="center"><code>'[-1.0]'</code></td>
      <td>operations to be performed per channel, as sorted by <code>channels</code>. Scalar to add to the complete field</td>
    </tr>
    <tr>
      <td><code>stdNorm</code></td>
      <td align="center"><code>'[1.0]'</code></td>
      <td>weighting for the division by the std of the first frame</td>
    </tr>
    <tr>
      <td><code>avgRemove</code></td>
      <td align="center"><code>'[0.0]'</code></td>
      <td>weighting of the average component removal</td>
    </tr>
    <tr>
      <td style="background-color:#cff0fa" colspan="3" align="center"><ins>Loss terms</ins> - <code>fooLambda</code>: weigthing for each loss term (set to 0 to disable).</td>
    </tr>
    <tr>
      <td><code>L2Lambda</code></td>
      <td align="center"><code>1.0</code></td>
      <td>Mean-squared error of channels</td>
    </tr>
    <tr>
      <td><code>GDLLambda</code></td>
      <td align="center"><code>1.0</code></td>
      <td>Mean-squared error of gradients of channels</td></tr>
    <tr>
      <td style="background-color:#a6e4f7" colspan="3" align="center"><b>Training monitoring</b></td>
    </tr>
    <tr>
      <td><code>freqToFile</code></td>
      <td align="center"><code>1</code></td>
      <td>epoch frequency for loss output to disk</td>
    </tr>
  </tbody>
</table>

Check the PyTorch [documentation](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) for more information on schedulers.

## Logger

Two loggers are available:
- `np`: saves losses in numpy format to disk (Default)
- `tb`: tensorboard logger

The user can specify it with arguments:

<pre style="background-color:#cff0fa"><code>python main.py --trainingConf configuration_train.yaml --logger tb
</code></pre>

## Testing the model

Script `test.py` contains the testing procedure implementation. It is a auto-regressive analysis, that is, previous estimations are used as input to advance in time. It does not reads the parameters of the training from the configuration files directly, so the properties must be given either on the code or in its configuration file. Test parameters are described next:

<table border="1" align="center">
  <tbody>
    <tr style="background-color:#82afbd">
      <td align="center"><b>parameter</b></td>
      <td align="center"><b>example</b></td>
      <td align="center"><b>description</b></td>
    </tr>
    <tr>
      <td><code>model_path</code></td>
      <td align="center"><code>1</code></td>
      <td>path to model folder, where the trained weigths are stored. Test data will be saved on 'results' subfolder</td>
    </tr>
    <tr>
      <td><code>reference_path</code></td>
      <td align="center"><code>/path/to/reference/{:06d}</code></td>
      <td>path to the reference solutions (must consider a format for the multiple scenes)</td>
    </tr>
    <tr>
      <td><code>file_format</code></td>
      <td align="center"><code>vtk{:06d}.vti1</code></td>
      <td>format of the reference simulation files</td>
    </tr>
    <tr>
      <td><code>comparison_name</code></td>
      <td align="center"><code>benchmarks</code></td>
      <td>name of the test to be performed, added to the generated files</td>
    </tr>
    <tr>
      <td><code>tester</code></td>
      <td align="center"><code>RecursiveTester</code></td>
      <td>name of tester class, defining with type of test. Check <code>test.py</code> and <code>pulsenetlib/tester.py</code> for more information.</td>
    </tr>
    <tr>
      <td><code>comparator</code></td>
      <td align="center"><code>ScalarComparator</code></td>
      <td>name of the comparator class, defining with type of comparisons between the reference and the estimated field to be performed.</td>
    </tr>
    <tr>
      <td><code>frame_step</code></td>
      <td align="center"><code>4</code></td>
      <td>frame step jump (in number of simulation timesteps)</td>
    </tr>
    <tr>
      <td><code>channel_index</code></td>
      <td align="center"><code>0</code></td>
      <td>index of the channel to analyze (plot and calculate statistics)</td>
    </tr>
    <tr>
      <td><code>frame_start</code></td>
      <td align="center"><code>0</code></td>
      <td>number of frame to start the analysis. A sufficient number of solution frames before this one is necessary. If 0, considers the first possible frame</td>
    </tr>
    <tr>
      <td><code>frames_to_estimate</code></td>
      <td align="center"><code>10</code></td>
      <td>number of frames that will be estimated, if 0 accounts for all possible cases from the timestesp in the reference folder. Equals the number of recurrencies plus one</td>
    </tr>
  </tbody>
</table>
    
Testing is perfomed on parallel in CPU using [mpi4py](https://mpi4py.readthedocs.io/en/stable/), a MPI library ([OpenMPI](https://www.open-mpi.org/), for example) must be available. It will consider every subfolder of `reference_path` named in numerical format as a scene (independent simulation), and uses the available timesteps first as input and after as the target fields. Command to run a test:

<pre style="background-color:#cff0fa"><code>mpirun -n &#60;NTASKS&#62; python test.py --test_config configuration_test.yaml 
</code></pre>

where `NTASKS` is the number of MPI processes. If the number of scenes is larger than the number of processes, the former are splitted as equally as possible among the processes.

Additional flags:

- `--export_field`: generate VTK files of the estimated fields
- `--no_panel`: not generating a panel with the contour plot of the target, estimation and error scalar fields
- `--no_graphs`: not generating the errors evolution graphs
- `--verbose`: display evolution of tests on terminal
- `--apply_EPC`: apply acoustic energy preservation correction (EPC) for each recurrence
