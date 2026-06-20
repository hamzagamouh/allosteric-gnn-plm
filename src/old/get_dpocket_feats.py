import os,subprocess

HOME_FOLDER="/storage/praha1/home/hamzagamouh"
inp_dir=f"{HOME_FOLDER}/allosteric/method_2/dpocket_feats"
# # RUN DPOCKET USING BINARY
DPOCKET_EXEC=f"{HOME_FOLDER}/fpocket_sandbox/usr/local/bin/dpocket"

def process_input(x):
    input_file=f"{inp_dir}/{x}"
    prefix=x.replace("mod.input","").replace(".pdb","").strip("_").replace("chain","")
    command=f"""
        mkdir {HOME_FOLDER}/allosteric/method_2/dpocket_feats/outputs/{prefix};
        cd {HOME_FOLDER}/allosteric/method_2/dpocket_feats/outputs/{prefix};
        {DPOCKET_EXEC} -f {input_file} -E -d 0.1;
        rm ./*fpocket*.txt;
        mv dpout_explicitp.txt {prefix}.txt;
        """
    # subprocess.call(command,
    #     shell=True
    # )
    return command

def is_processed(x):
    prefix=x.replace("mod.input","").replace(".pdb","").strip("_").replace("chain","")
    return os.path.exists(f"{HOME_FOLDER}/allosteric/method_2/dpocket_feats/outputs/{prefix}/{prefix}.txt")

def get_new_inputs():
    inps=os.listdir(f"{HOME_FOLDER}/allosteric/method_2/dpocket_feats")
    return [x for x in inps if not is_processed(x) and x.endswith(".input")]

def process_available():
    inputs=get_new_inputs()
    if len(inputs)>0:
        print(f'Processing {len(inputs)} inputs')
        for x in inputs:
            if not is_processed(x):
                process_input(x)
        inputs=get_new_inputs()

def get_next_commands(inputs):
    commands=[]
    for x in inputs:
        if not is_processed(x):
            commands+=[process_input(x)]
            if len(commands)>=os.cpu_count():
                return commands

# Start all subprocesses

inputs=get_new_inputs()
while len(inputs)>0:
    commands=get_next_commands(inputs)
    processes = [subprocess.Popen(cmd,shell=True) for cmd in commands]

    # # Wait for all to complete
    for p in processes:
        p.wait()
    inputs=get_new_inputs()