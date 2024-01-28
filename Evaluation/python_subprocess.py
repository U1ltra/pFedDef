import subprocess

EXP_PATH = input("Enter the path to the experiment: ")
CODE_PATH = "/home/ubuntu/Documents/jiarui/pFedDef"

while True:
    result = subprocess.run(
        f"echo {EXP_PATH} | python {CODE_PATH}/eval_acc_io_path.py",
        shell=True,
        capture_output=True,
        text=True,
    )
    print(result.stdout)

    if result.returncode != 0:
        print("Error:", result.stderr)
    else:
        print("Evaluation success!")
        break
