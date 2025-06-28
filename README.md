# SKOS Converter

A bidirectional converter between SKOS (Simple Knowledge Organization System) RDF vocabularies and common formats for visual editors like Notion. This tool enables knowledge engineers to manage controlled vocabularies and thesauri in a user-friendly interface while maintaining standards-compliant SKOS data.

## Features

- **SKOS → Notion**: Convert SKOS Turtle files to Notion-compatible formats (Markdown, CSV, JSON)
- **Notion → SKOS**: Convert Notion markdown exports back to SKOS Turtle
- **Validation**: Comprehensive SKOS validation with error and warning detection
- **Hierarchy Preservation**: Maintains broader/narrower relationships and concept schemes
- **Metadata Support**: Preserves definitions, alternative labels, notations, and URIs
- **UUID Generation**: Automatically creates unique identifiers for new concepts
- **Circular Reference Detection**: Identifies and handles circular relationships
- **Multiple Output Formats**: Choose between CSV, Markdown, or JSON for Notion import

## Installation

### Requirements
- Python 3.6 or higher
- rdflib library

### Setup
```bash
# Clone or download the script
wget https://raw.githubusercontent.com/yourusername/skos-notion-converter/main/skos-to-notion-converter.py

# Install dependencies
pip install rdflib

# Make executable (optional)
chmod +x skos-to-notion-converter.py
```

## Usage

The converter uses subcommands for different conversion directions:

### Converting SKOS to Notion

```bash
# Basic conversion to CSV (default)
python3 skos-to-notion-converter.py to-notion vocabulary.ttl

# Convert to Markdown with visual formatting
python3 skos-to-notion-converter.py to-notion vocabulary.ttl --format markdown

# Convert to all formats
python3 skos-to-notion-converter.py to-notion vocabulary.ttl --format all

# Skip validation
python3 skos-to-notion-converter.py to-notion vocabulary.ttl --skip-validation

# Force conversion despite errors
python3 skos-to-notion-converter.py to-notion vocabulary.ttl --force

# Custom output name
python3 skos-to-notion-converter.py to-notion vocabulary.ttl --output my_vocab --format all
```

### Converting Notion to SKOS

```bash
# Basic conversion with default namespace
python3 skos-to-notion-converter.py to-skos notion_export.md

# Specify custom namespace and prefix
python3 skos-to-notion-converter.py to-skos notion_export.md \
  --namespace "http://data.example.com/vocab#" \
  --prefix "ex"

# Custom output file
python3 skos-to-notion-converter.py to-skos notion_export.md --output my_vocab.ttl
```

### Getting Help

```bash
# General help
python3 skos-to-notion-converter.py --help

# Help for specific subcommand
python3 skos-to-notion-converter.py to-notion --help
python3 skos-to-notion-converter.py to-skos --help
```

## SKOS Validation

The converter performs comprehensive validation checking for:

### Critical Errors (block conversion by default)
- **Duplicate URIs**: Same URI used for multiple resources
- **Missing labels**: Concepts without prefLabel or rdfs:label
- **Circular references**: Concepts that reference each other in loops
- **Multiple preferred labels**: More than one prefLabel per language on a concept

### Warnings (valid SKOS but worth attention)
- **Duplicate label text**: Same label used by different concepts
- **Orphan concepts**: Concepts without broader relations or top concept status
- **Missing concept schemes**: Concepts not associated with any scheme
- **Polyhierarchy**: Concepts with multiple broader concepts
- **Deep hierarchies**: Hierarchies deeper than 7 levels

Use `--force` to convert despite critical errors, or `--skip-validation` to bypass validation entirely.

## Notion Import/Export Guide

### Importing to Notion

#### From CSV:
1. In Notion, create a new page
2. Type `/table` and select "Table - Full page"
3. Click "..." menu → "Import" → "CSV"
4. Select your generated CSV file
5. Configure the database:
   - Set "Title" as the page title property
   - Use "Parent" property to create linked relations
   - Group by "Concept Scheme" or filter by "Level"
   - Add visual indicators based on "Level" for hierarchy

#### From Markdown:
1. Create a new Notion page
2. Copy the entire markdown content
3. Paste into Notion (Cmd/Ctrl+V)
4. Optional: Convert to toggle lists:
   - Select hierarchical sections
   - Press Cmd/Ctrl+Shift+7
5. The visual indicators (▸ ▹ ◦) help show hierarchy depth

### Exporting from Notion

1. Open your Notion vocabulary page
2. Click "..." menu → "Export"
3. Choose "Markdown & CSV"
4. Include subpages: "Everything"
5. Extract the markdown file from the zip

## Notion Markdown Format for SKOS

When creating vocabularies in Notion for SKOS export, follow this structure:

```markdown
# Concept Scheme: My Vocabulary

## First Top Concept
_Definition:_ Description of this concept
_Alternative Labels:_ Synonym1, Synonym2
_Notation:_ CODE1
<sub>URI: http://example.org/vocab#concept1</sub>

### Narrower Concept 1
_Definition:_ Description of narrower concept

### Narrower Concept 2
_Definition:_ Another narrower concept

## Second Top Concept
_Definition:_ Description of second top concept
```

### Formatting Rules:
- **H1 (`#`)**: Concept Schemes
- **H2 (`##`)**: Top Concepts
- **H3+ (`###`, etc.)**: Narrower concepts (hierarchy based on heading level)
- **Metadata** (all optional):
  - `_Definition:_` or `**Definition:**` → skos:definition
  - `_Alternative Labels:_` → skos:altLabel (comma-separated)
  - `_Notation:_` → skos:notation
  - `<sub>URI: ...</sub>` → Preserves existing URI

### Important Notes:
- If no definition is provided, "Lorem ipsum" will be used as placeholder
- New concepts without URIs will be assigned UUID-based identifiers
- Existing URIs in the markdown will be preserved during round-trip conversion
- All concepts automatically get skos:inScheme relationship to their scheme

## Output Format Examples

### Markdown Output Features:
- **Visual hierarchy indicators**: ▸ ▹ ◦ for different depths
- **Icons**: 📂 for schemes, 📁 for grouped sections, 📄 for unassigned
- **Table of Contents**: Auto-generated with links
- **Metadata formatting**: Italicized with inline display
- **URIs**: Shown in small text for reference

### CSV Output Structure:
- **Title**: Concept label with indentation for hierarchy
- **Parent**: Parent concept for hierarchy building
- **Concept Scheme**: Associated scheme name
- **Definition**: Concept definition
- **Alternative Labels**: Comma-separated list
- **Notation**: Code or identifier
- **URI**: Original URI
- **Level**: Numeric depth in hierarchy

### JSON Output:
Structured for Notion API integration with full hierarchy and relationships preserved.

## Example Workflows

### Workflow 1: Create New Vocabulary in Notion
1. Structure your vocabulary in Notion using the markdown format
2. Export from Notion as markdown
3. Convert to SKOS: 
   ```bash
   python3 skos-to-notion-converter.py to-skos export.md \
     --namespace "http://my-domain.com/vocab#" \
     --prefix "myprefix"
   ```
4. Use the generated .ttl file in your semantic web applications

### Workflow 2: Edit Existing SKOS Vocabulary
1. Convert SKOS to Notion: 
   ```bash
   python3 skos-to-notion-converter.py to-notion vocab.ttl --format markdown
   ```
2. Import markdown to Notion
3. Edit in Notion (add concepts, reorganize hierarchy, update definitions)
4. Export from Notion
5. Convert back: 
   ```bash
   python3 skos-to-notion-converter.py to-skos edited_export.md \
     --namespace "http://original-namespace.com/vocab#"
   ```

### Workflow 3: Team Collaboration
1. Convert SKOS to CSV: 
   ```bash
   python3 skos-to-notion-converter.py to-notion vocab.ttl --format csv
   ```
2. Import as Notion database
3. Share with team for collaborative editing
4. Use database features:
   - Comments on concepts
   - Filters by scheme or level
   - Sort by various properties
   - Create different views
5. Export and convert back to SKOS when ready

## Best Practices

### For SKOS Files:
1. **Validate first**: Always run without `--skip-validation` initially
2. **Fix circular references**: These can cause infinite loops
3. **Ensure unique URIs**: Duplicate URIs will cause data corruption
4. **Use proper escaping**: Periods and special characters in quoted strings are fine

### For Notion:
1. **Consistent formatting**: Use the exact metadata format shown
2. **Meaningful definitions**: Avoid relying on "Lorem ipsum" placeholders
3. **Preserve URIs**: Include `<sub>URI: ...</sub>` to maintain identifiers
4. **Check hierarchy**: Ensure heading levels correctly represent relationships

### For Round-trip Conversion:
1. **Namespace consistency**: Use the same namespace when converting back
2. **URI preservation**: Original URIs are maintained if included in markdown
3. **Regular backups**: Keep copies of both SKOS and Notion versions
4. **Version control**: Track changes in your vocabulary over time

## Troubleshooting

### Common Issues:

**"Bad syntax" error when parsing Turtle**
- Check for missing periods at end of statements
- Verify all URIs are enclosed in angle brackets
- Ensure quotes are properly closed
- Validate at http://ttl.summerofcode.be/

**"Maximum recursion depth exceeded"**
- Your SKOS file has circular references
- Run validation to identify the cycle
- Fix the circular broader/narrower relationships

**Empty output file**
- Check that markdown has proper heading structure
- Ensure H1 exists before H2 concepts
- Verify the file encoding is UTF-8

**Missing concepts after conversion**
- Check for duplicate URIs in source data
- Verify all concepts have required properties
- Review validation warnings

**"EOL while scanning string literal"**
- This is a Python syntax error in the script itself
- Ensure the regex pattern on line 775 is: `re.match(r'^(#+)\s+(.+)$', line)`
- The pattern must include the closing `$'` and `, line)`

### Debug Mode:
The converter includes debug output showing:
- Python version and arguments
- File paths and parsing progress
- Number of triples/concepts processed
- Validation results

## Limitations

1. **Language tags**: Currently doesn't preserve language tags on labels
2. **Complex SKOS features**: Some advanced SKOS features like collections are not supported
3. **Notion limitations**: 
   - Maximum 6 heading levels (deeper hierarchies use indentation)
   - Toggle list conversion must be done manually
4. **Round-trip fidelity**: Some SKOS properties may be lost in conversion

## Contributing

Contributions are welcome! Please submit issues and pull requests on GitHub.

### Areas for Improvement:
- Language tag support
- SKOS collections and ordered collections
- Additional validation rules
- Notion API direct integration
- Batch processing multiple files

## License

CC0

## Acknowledgments

Built with [RDFLib](https://rdflib.readthedocs.io/) for RDF processing.
Built with assistance from Claude.ai

## Version History

- 1.0.0: Initial release with bidirectional conversion
- 1.1.0: Added validation and circular reference detection
- 1.2.0: Improved hierarchy handling and visual formatting

---

For questions or support, please open an issue on the GitHub repository.
