#!/bin/bash
ps aux|grep chatchat|grep -v grep|awk '{print $2}'|xargs kill -9
