#!/bin/sh
../../sbin/stop-dfs.sh
mkdir -p ../../mynode/namenode
mkdir -p ../../mynode/datanode
rm -r ../../mynode/namenode/*
rm -r ../../mynode/datanode/*
../../bin/hdfs namenode -format
../../sbin/start-dfs.sh