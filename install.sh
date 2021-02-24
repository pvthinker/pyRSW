#!/bin/bash

pydir=$HOME/.pyrsw
default=defaults.yaml

echo "--------------------------------------------------------------------------------"
echo ""
echo " Installing pyRSW"
echo ""
if [ ! -d "$pydir" ]; then
    echo "Create $pydir"
    mkdir $pydir
fi
if [ ! -f "$pydir/$default" ]; then
    echo "Copy $default in $pydir"
    cp $default $pydir/
fi
echo ""
echo "  To use pyRSW you simply need to enter"
echo "     source ~/.pyrsw/activate.sh  if you're under bash"
echo "  or source ~/.pyrsw/activate.csh  if you're under csh/tcsh"
echo ""
echo "  However, before starting, please read this note carefully"
echo ""
echo "  As it configured, pyRSW will store the results in"
echo ""
echo "      *** $HOME/data/pyrsw ***"
echo ""
echo "  If you don't run the code from your laptop then it is likely that"
echo "  you are not allowed to store large binary files on $HOME/data"
echo "  because this is your home."
echo ""
echo "  In this case, please, edit $HOME/.pyrsw/defaults.yaml"
echo "  And set 'datadir' (in the output section) default value to a"
echo "  place where you are authorized to store large binary files."
echo "  In the jargon, this place is usually a 'work' directory."
echo ""
echo "  If you don't know where to store your results check that"
echo "  with your system administrator."
echo ""
echo "  Once you know where to store the results, then the fun starts!"
echo ""

# for bash users
cat > $pydir/activate.sh << EOF
export PYTHONPATH=`pwd`/core
echo Python now knows that pyRSW is in `pwd`
EOF

# for csh, tcsh users
cat > $pydir/activate.csh << EOF
setenv PYTHONPATH `pwd`/core
echo Python now knows that pyRSW is in `pwd`
EOF
