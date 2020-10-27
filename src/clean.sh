#!/bin/bash

sed "s/i32//g" $1.dirty | sed "s/f32.nan/nan/g" | sed "s/f32//g" > "$1.clean"
