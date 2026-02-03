#!/usr/bin/env python3
import re, json, ast, argparse, sys

def parse_shape_string(shape_str: str):
    """
    Converts a shape string '[80, 64, 768]' into a list [80, 64, 768].
    """
    try:
        parsed_shape = ast.literal_eval(shape_str)
        if isinstance(parsed_shape, (list, tuple)):
            return list(parsed_shape)
        return None
    except (ValueError, SyntaxError):
        return None

def parse_number_string(num_str: str) -> int:
    """
    Converts a number string with commas '3,087,790,080' to an integer.
    Handles '--' as 0.
    """
    if num_str.strip() == '--':
        return 0
    return int(num_str.replace(',', ''))

def parse_model_summary_hierarchical(file_path: str) -> list:
    """
    Parses the model summary file, extracts all operators in order of appearance,
    and generates a 'Fully Qualified Name' (FQN) for each.
    """
    
    operator_test_cases = [] 
    layer_name_regex = re.compile(r'([A-Za-z0-9_]+)\s*\(([a-zA-Z0-9_]+)\)')
    prefix_regex = re.compile(r'^[│└├\s-]*')
    line_regex = re.compile(
        r'^(?P<layer_info>.*?)\s+'
        r'(?P<input_shape>\[.*?\]|--)\s+'
        r'(?P<output_shape>\[.*?\]|--)\s+'
        r'(?P<params>[\d,]+|--)\s+'
        r'(?P<mult_adds>[\d,]+|--)$'
    )

    indent_stack = [-1] 
    path_stack = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('===') or line.startswith('Layer (type'):
                continue
            line_str = line.strip()
            if not line_str:
                continue

            prefix_match = prefix_regex.match(line)
            current_indent = len(prefix_match.group(0))
            line_match = line_regex.match(line_str)
            if not line_match:
                continue
                
            layer_info_str = line_match.group('layer_info')
            name_match = layer_name_regex.search(layer_info_str)
            if not name_match:
                continue

            op_type = name_match.group(1)
            local_name = name_match.group(2)

            while current_indent <= indent_stack[-1]:
                indent_stack.pop()
                path_stack.pop()

            path_stack.append(local_name)
            indent_stack.append(current_indent)
            
            mult_adds_str = line_match.group('mult_adds').strip()
            if mult_adds_str == '--' or parse_number_string(mult_adds_str) == 0:
                continue

            if len(path_stack) > 1:
                fully_qualified_name = ".".join(path_stack[1:])
            else:
                fully_qualified_name = local_name

            input_shape_str = line_match.group('input_shape').strip()
            output_shape_str = line_match.group('output_shape').strip()
            input_shape = parse_shape_string(input_shape_str)
            output_shape = parse_shape_string(output_shape_str)
            mult_adds = parse_number_string(mult_adds_str)
            
            operator_test_cases.append({
                "fully_qualified_name": fully_qualified_name,
                "op_type": op_type,
                "local_name": local_name,
                "input_shape": input_shape,
                "output_shape": output_shape,
                "mult_adds": mult_adds
            })

    return operator_test_cases

def main():
    """
    Responsible for converting the model summary to test cases (JSON) only.
    """
    parser = argparse.ArgumentParser(
        description="Parses model summary (txt) into operator test cases (json).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i", "--input-file",
        default="model_summary",
        help="Path to the input model summary file."
    )
    parser.add_argument(
        "-o", "--output-file",
        default="operator_test_cases.json",
        help="Path to the output operator test cases (JSON) file."
    )
    
    args = parser.parse_args()

    try:
        print(f"--- 1. Parsing Phase ---")
        print(f"Parsing {args.input_file} ...")
        
        test_cases = parse_model_summary_hierarchical(args.input_file)
        
        print(f"Parsing complete! Found {len(test_cases)} operator instances in sequential order.")

        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(test_cases, f, indent=4)
            
        print(f"Test cases saved to: {args.output_file}")

    except FileNotFoundError:
        print(f"Error: File {args.input_file} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during parsing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
