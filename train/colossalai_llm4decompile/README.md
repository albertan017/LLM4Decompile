## Guide for Training the Model

We recommend using the [Colossal AI](https://github.com/hpcaitech/ColossalAI) framework to train **LLM4Decompile**. For more details, please refer to the [Colossal-LLaMA](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Colossal-LLaMA) application documentation.

Please follow the official Colossal AI tutorial to set up the training environment.

Here are some lessons based on our experience:
1. Manually install `xentropy, layer_norm, and rotary`, and update `flash-attention` to the latest version.
2. Install the [Nvidia apex](https://github.com/NVIDIA/apex) library manually to leverage mixed precision training. Accelerate the installation process using Ninja by `export MAX_JOBS=128`.
3. It seems that versions of numpy above 2.0 have compatibility issues. Please downgrade the numpy version by running `pip install numpy==1.23.3`.
4. Ensure your SSH service is configured and running correctly.
5. Utilize the TensorBoard plugin in Visual Studio Code for convenient monitoring of training loss.

Once the environment is properly configured, initiate the training process by executing:
```bash
bash run_llm4decompile_train.sh
```