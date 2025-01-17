#!/bin/bash
export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
asetup Athena,main,latest
export X509_USER_PROXY=/data/fpiazza/ggHyyd/Test/x509up_u14828
lsetup "rucio -w"
export X509_USER_PROXY=/data/fpiazza/ggHyyd/Test/x509up_u14828
python /data/fpiazza/ggHyyd/Test/merge.py $1 /data/fpiazza/ggHyyd/Test/MC23d_xsecs grid $3 $2 /data/fpiazza/ggHyyd/Test/rundir
