# Runs the two long trainings one after the other, so they do not fight over the processor.
# Each writes a log you can watch while it runs.
$root = Split-Path -Parent $MyInvocation.MyCommand.Path

Set-Location "$root\5_story_gpt"
python -u train.py 6.5 *> train_log.txt

Set-Location "$root\4_diffusion"
python -u train_mnist.py 4000 *> train_log_mnist.txt
