import json
import xml.etree.ElementTree as ET
from argparse import ArgumentParser

from utils import tqdm, decorator


@decorator(print, 'loading')
def load_case_frame_dict(input_path: str) -> ET.Element:
    return ET.parse(input_path).getroot()


def main():
    parser = ArgumentParser()
    parser.add_argument('INPUT', type=str, help='path to input')
    parser.add_argument('OUTPUT', type=str, help='path to output')
    args = parser.parse_args()

    root = load_case_frame_dict(args.INPUT)

    converted = {}
    for entry in tqdm(root, dynamic_ncols=True):
        predicate_total = 0.
        predicate = f'{entry.attrib["headword"]}:{entry.attrib["predtype"]}'
        converted[predicate] = {}

        for case_frame in entry:
            # case_frame.attrib["id"] ... e.g. '入札/にゅうさつ+する/する+れる/れる:動1'
            case_frame_total = 0.
            converted[predicate][case_frame.attrib["id"]] = {}

            for case in case_frame:
                # case.attrib ... e.g. {'case': 'ガ格', 'frequency': '5'}
                case_total = 0.
                converted[predicate][case_frame.attrib["id"]][case.attrib["case"]] = {}

                for argument in case:
                    # argument.attrib ... {'frequency': '1'}
                    # argument.text ... e.g. '誰/だれ'
                    converted[predicate][case_frame.attrib["id"]][case.attrib["case"]][argument.text] = float(
                        argument.attrib["frequency"]
                    )  # 0.667などもあるのでfloat
                    case_total += float(argument.attrib["frequency"])
                else:
                    converted[predicate][case_frame.attrib["id"]][case.attrib["case"]]['case_total'] = case_total

                case_frame_total += float(case.attrib["frequency"])
                predicate_total += float(case.attrib["frequency"])
            else:
                converted[predicate][case_frame.attrib["id"]]['case_frame_total'] = case_frame_total
        else:
            converted[predicate]['predicate_total'] = predicate_total

    with open(args.OUTPUT, mode='w') as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()