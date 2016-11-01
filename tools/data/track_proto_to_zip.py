#!/usr/bin/env python
import argparse

import sys
import os.path as osp
this_dir = osp.dirname(__file__)
sys.path.insert(0, osp.join(this_dir, '../../src/'))
sys.path.insert(0, osp.join(this_dir, '../../external/'))
from vdetlib.utils.protocol import proto_load
from tpn.data_io import save_track_proto_to_zip

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('proto_file')
    parser.add_argument('save_zip')
    args = parser.parse_args()

    proto = proto_load(args.proto_file)
    save_track_proto_to_zip(proto, args.save_zip)

