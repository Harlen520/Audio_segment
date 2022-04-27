import os
import subprocess
import csv
import argparse
from collections import defaultdict
import json
from tqdm import tqdm
from intervaltree import IntervalTree
from dcase_util.containers import MetaDataContainer, MetaDataItem


BAT_MAPPING = {'1_Music': 'music',
               '2_FgMusic': 'fg-music',
               '3_Similar': 'similar',
               '4_BgMusic': 'bg-music',
               '5_BgMusicVL': 'bgvl-music',
               '6_NoMusic': 'no-music'}
MAPPINGS = ['MD', 'MRLE']
MD_MAPPING = {'music': 'music',
              'fg-music': 'music',
              'similar': 'music',
              'bg-music': 'music',
              'bgvl-music': 'music',
              'no-music': 'no-music'}
MRLE_MAPPING = {'music': 'fg-music',
                'fg-music': 'fg-music',
                'similar': 'bg-music',
                'bg-music': 'bg-music',
                'bgvl-music': 'bg-music',
                'no-music': 'no-music'}


# *************************
# Generation of annotations
# *************************


def _get_interval_class(class_):
        return BAT_MAPPING[class_]


def _generate_intervals(its_dict, bat_csv_path):
    with open(bat_csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            (_, wav, start, end, class_, _, _) = row
            if wav not in its_dict.keys():
                its_dict[wav] = IntervalTree()
            class_ = _get_interval_class(class_)
            if class_ is None:
                continue
            its_dict[wav].addi(round(float(start), 2),
                               round(float(end), 2),
                               class_)


def _generate_original_annotations(bat_csv_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    its_dict = {}
    _generate_intervals(its_dict, bat_csv_path)
    line = '{0}\t{1}\t{2}\n'
    for wav, its in its_dict.items():
        its = sorted(its)
        output_fname = os.path.join(output_dir, wav.replace('.wav', '.tsv'))
        with open(output_fname, 'w') as f:
            last_start, last_end, last_class = its[0]
            if len(its) == 1:
                f.write(line.format(last_start, last_end, last_class))
                continue
            for curr_start, curr_end, curr_class in its[1:]:
                if curr_class != last_class:
                    f.write(line.format(last_start, curr_start, last_class))
                    last_start = curr_start
                    last_class = curr_class
            f.write(line.format(last_start, curr_end, curr_class))


def _mapping(tsv_original_annotator_dir, tsv_mapped_annotator_dir, mapping):
    for fname in os.listdir(tsv_original_annotator_dir):
        original_fpath = os.path.join(tsv_original_annotator_dir, fname)
        with open(original_fpath, 'r') as f:
            rows = f.readlines()
        mapped_fpath = os.path.join(tsv_mapped_annotator_dir, fname)
        with open(mapped_fpath, 'w') as f:
            last_start, last_end, last_class = rows[0].split('\t')
            last_class = mapping[last_class.replace('\n', '')]
            if len(rows) == 1:
                f.write('\t'.join([last_start, last_end, last_class + '\n']))
                continue
            for row in rows[1:]:
                curr_start, curr_end, curr_class = row.split('\t')
                curr_class = mapping[curr_class.replace('\n', '')]
                if curr_class != last_class:
                    f.write('\t'.join([last_start,
                                       curr_start,
                                       last_class + '\n']))
                    last_start = curr_start
                    last_class = curr_class
            f.write('\t'.join([last_start, curr_end, curr_class + '\n']))


def _apply_mapping(tsv_original_dir, tsv_mapped_dir, mapping):
    mapping = MD_MAPPING if mapping == 'MD' else MRLE_MAPPING
    if not os.path.exists(tsv_mapped_dir):
        os.makedirs(tsv_mapped_dir)
    for annotator in os.listdir(tsv_original_dir):
        tsv_original_annotator_dir = os.path.join(tsv_original_dir, annotator)
        tsv_mapped_annotator_dir = os.path.join(tsv_mapped_dir, annotator)
        if not os.path.exists(tsv_mapped_annotator_dir):
            os.makedirs(tsv_mapped_annotator_dir)
        _mapping(tsv_original_annotator_dir, tsv_mapped_annotator_dir, mapping)


def _feed_annotations_to_dict(d, tsv_dir):
    agreement_d = {}
    subdirs = os.listdir(tsv_dir)
    for sd in subdirs:
        d['annotations'][sd] = {}
        sd_path = os.path.join(tsv_dir, sd)
        for fname in os.listdir(sd_path):
            with open(os.path.join(sd_path, fname), 'r') as f:
                rows = f.readlines()
            fname = fname.replace('.tsv', '')
            d['annotations'][sd][fname] = {}
            if fname not in agreement_d:
                agreement_d[fname] = IntervalTree()
            for i, row in enumerate(rows):
                start, end, class_ = row.split('\t')
                start = float(start)
                end = float(end)
                class_ = class_.replace('\n', '')
                d['annotations'][sd][fname][i] = {}
                d['annotations'][sd][fname][i]['start'] = start
                d['annotations'][sd][fname][i]['end'] = end
                d['annotations'][sd][fname][i]['class'] = class_
                agreement_d[fname].addi(start, end, '__'.join([class_, sd]))
    return agreement_d


def _feed_agreement_to_dict(d, agreement_d, tsv_dir):
    d['agreement'] = {}
    subdirs = os.listdir(tsv_dir)
    for fname in agreement_d.keys():
        d['agreement'][fname] = {}
        d['agreement'][fname]['vals'] = {}
        d['agreement'][fname]['segs'] = {}
        d['agreement'][fname]['segs']['full'] = []
        d['agreement'][fname]['segs']['partial'] = []
        agreement_d[fname].split_overlaps()
        used_intervals = []
        duration = 0.0
        full_agreement = 0.0
        partial_agreement = 0.0
        for it in sorted(agreement_d[fname]):
            interval = (it[0], it[1])
            if interval not in used_intervals:
                used_intervals.append(interval)
            else:
                continue
            duration += it[1] - it[0]
            found_its = agreement_d[fname].overlap(it[0], it[1])
            if len(found_its) != len(subdirs):
                raise ValueError("Missing annotation")
            classes = defaultdict(int)
            for it_ in found_its:
                class_ = it_[2].split('__')[0]
                classes[class_] += 1
                # if class_ not in classes:
                #     classes.append(class_)
            if len(classes) == 1:
                full_agreement += it[1] - it[0]
                d['agreement'][fname]['segs']['full'].append([it[0], it[1], list(classes.keys())[0]])
                partial_agreement += it[1] - it[0]
                d['agreement'][fname]['segs']['partial'].append([it[0], it[1], list(classes.keys())[0]])
            elif len(classes) == 2:
                partial_agreement += it[1] - it[0]
                cls1, cls2 = list(classes.keys())
                if classes[cls1] > classes[cls2]:
                    d['agreement'][fname]['segs']['partial'].append([it[0], it[1], cls1])
                else:
                    d['agreement'][fname]['segs']['partial'].append([it[0], it[1], cls2])
        d['agreement'][fname]['vals']['full'] = full_agreement / duration
        d['agreement'][fname]['vals']['partial'] = partial_agreement / duration


def _merge_agreement_segments(d, agreement_d):
    for fname in agreement_d.keys():
        fa_segs = sorted(d['agreement'][fname]['segs']['full'])
        pa_segs = sorted(d['agreement'][fname]['segs']['partial'])
        if fa_segs != []:
            merged_fa_segs = []
            last_fa_seg = fa_segs[0]
            last_fa_start = fa_segs[0][0]
            for s in fa_segs[1:]:
                if s[0] != last_fa_seg[1]:
                    merged_fa_segs.append([last_fa_start, last_fa_seg[1], last_fa_seg[2]])
                    last_fa_start = s[0]
                elif last_fa_seg[2] != s[2]:
                    merged_fa_segs.append([last_fa_start, last_fa_seg[1], last_fa_seg[2]])
                    last_fa_start = s[0]
                last_fa_seg = s
            merged_fa_segs.append([last_fa_start, last_fa_seg[1], last_fa_seg[2]])
            d['agreement'][fname]['segs']['full'] = sorted(merged_fa_segs)
        if pa_segs != []:
            merged_pa_segs = []
            last_pa_seg = pa_segs[0]
            last_pa_start = pa_segs[0][0]
            for s in pa_segs[1:]:
                if s[0] != last_pa_seg[1]:
                    merged_pa_segs.append([last_pa_start, last_pa_seg[1], last_fa_seg[2]])
                    last_pa_start = s[0]
                elif last_fa_seg[2] != s[2]:
                    merged_fa_segs.append([last_fa_start, last_fa_seg[1], last_fa_seg[2]])
                    last_fa_start = s[0]
                last_pa_seg = s
            merged_pa_segs.append([last_pa_start, last_pa_seg[1], last_fa_seg[2]])
            d['agreement'][fname]['segs']['partial'] = sorted(merged_pa_segs)

def _convert_annotations_to_json(tsv_dir, json_path):
    d = {}
    d['annotations'] = {}
    # Ingest annotations into the dictionary and feed the agreement IT
    agreement_d = _feed_annotations_to_dict(d, tsv_dir)
    # Ingest full and partial agreement percentages for each audio file
    # and the lists of segments with full and partial agreement
    _feed_agreement_to_dict(d, agreement_d, tsv_dir)
    # Merge contiguous full and partial agreement segments
    _merge_agreement_segments(d, agreement_d)
    # Save dict to JSON file
    with open(json_path, 'w') as f:
        json.dump(d, f)


def generate_annotations(bat_csv_dir, output_dir):
    # Generate the annotations by file and annotator for the complete taxonomy
    tsv_original_dir = os.path.join(output_dir, 'tsv', 'original')
    if not os.path.exists(tsv_original_dir):
        os.makedirs(tsv_original_dir)
    for bat_annotations_csv in os.listdir(bat_csv_dir):
        annotator_dir = os.path.splitext(bat_annotations_csv)[0]
        annotator_dir = os.path.join(tsv_original_dir, annotator_dir)
        if not os.path.exists(annotator_dir):
            os.mkdir(annotator_dir)
        bat_csv_path = os.path.join(bat_csv_dir, bat_annotations_csv)
        _generate_original_annotations(bat_csv_path, annotator_dir)
    # Map the annotations by file and annotator
    tsv_MD_dir = os.path.join(output_dir, 'tsv', 'MD_mapping')
    tsv_MRLE_dir = os.path.join(output_dir, 'tsv', 'MRLE_mapping')
    _apply_mapping(tsv_original_dir, tsv_MD_dir, 'MD')
    _apply_mapping(tsv_original_dir, tsv_MRLE_dir, 'MRLE')
    # Generate JSON files
    json_dir = os.path.join(output_dir, 'json')
    if not os.path.exists(json_dir):
        os.mkdir(json_dir)
    json_original = os.path.join(json_dir, 'original.json')
    json_MD = os.path.join(json_dir, 'MD_mapping.json')
    json_MRLE = os.path.join(json_dir, 'MRLE_mapping.json')
    _convert_annotations_to_json(tsv_original_dir, json_original)
    _convert_annotations_to_json(tsv_MD_dir, json_MD)
    _convert_annotations_to_json(tsv_MRLE_dir, json_MRLE)


# ***************************************
# Cut full and partial agreement excerpts
# ***************************************


def cut_audio_with_agreement(agr_level, json_path, audio_dir, output_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)
    for fname in tqdm(data['agreement'].keys()):
        segs = data['agreement'][fname]['segs'][agr_level]
        fpath = os.path.join(audio_dir, fname + '.wav')
        for s in segs:
            output_fname = '_'.join([fname, str(s[0]), str(s[1]), s[2]]) + '.wav'
            output_fpath = os.path.join(output_dir, output_fname)
            cmd = 'ffmpeg -loglevel panic -i {0} -ss {1} -to {2} {3}'.format(
                                                             fpath, s[0], s[1],
                                                             output_fpath)
            subprocess.call(cmd.split(' '))
           


# *****************************************
# Load annotations as dcase_util containers
# *****************************************


def load_annotations_as_containers(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    annotations = {}
    for annotator in data['annotations'].keys():
        annotations[annotator] = MetaDataContainer()
        for fname in data['annotations'][annotator].keys():
            segments = data['annotations'][annotator][fname].values()
            for s in segments:
                item = MetaDataItem({
                    'filename': fname,
                    'onset': s['start'],
                    'offset': s['end'],
                    'event_label': s['class']
                })
                annotations[annotator].append(item)
    return annotations
