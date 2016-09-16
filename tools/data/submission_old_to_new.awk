#!/usr/bin/env awk -f

{
    for (i = 1; i <= NF; i++) {
        if (i == 3) printf "-1 ";
        printf $i" "
    }
    printf "\n"
}