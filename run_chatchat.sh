#!/bin/bash
cd /root/work/project/langchain-ChatGLM
source activate chatchat
nohup python startup.py -a >> /root/work/project/langchain-ChatGLM/logs_my/logs_run_server-`date +%Y%m%d`.log 2>&1 &
