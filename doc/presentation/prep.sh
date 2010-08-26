#!/bin/bash

xrandr --output LVDS1 --mode 1024x768
xrandr --output VGA1 --mode 1024x768 --left-of LVDS1
