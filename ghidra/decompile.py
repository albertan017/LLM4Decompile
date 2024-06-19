#!/usr/bin/env python2
# -*- coding:utf-8 -*-

"""
Python Script used to communicate with Ghidra's API.
It will decompile all the functions of a defined binary and
save results into decompiled_output.c

The code is pretty straightforward, it includes comments and it is easy to understand.
This will help people that is starting with Automated Malware Analysis
using Headless scripts with Ghidra.

Modified from https://github.com/galoget/ghidra-headless-scripts
"""

import sys
from ghidra.app.decompiler import DecompInterface
from ghidra.util.task import ConsoleTaskMonitor
import __main__ as ghidra_app
args = ghidra_app.getScriptArgs()

# Communicates with Decompiler Interface
decompinterface = DecompInterface()

# Open Current Program
decompinterface.openProgram(currentProgram);

# Get Binary Functions
functions = currentProgram.getFunctionManager().getFunctions(True)

# Prints Current Python version (2.7)
print "Current Python version: " + str(sys.version.decode())

# Iterates through all functions in the binary and decompiles them
# Then prints the Pseudo C Code

with open(args[0], "w") as output_file:
    for function in list(functions):
        # Add a comment with the name of the function
        # print "// Function: " + str(function)
        output_file.write("// Function: " + str(function))

        # Decompile each function
        decompiled_function = decompinterface.decompileFunction(function, 0, ConsoleTaskMonitor())
        # Print Decompiled Code
        # print decompiled_function.getDecompiledFunction().getC()
        output_file.write(decompiled_function.getDecompiledFunction().getC())
