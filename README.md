# SKOS-Notion Converter

A bidirectional converter between SKOS (Simple Knowledge Organization System) RDF vocabularies and Notion-compatible formats. This tool enables knowledge engineers to manage controlled vocabularies and thesauri in Notion's user-friendly interface while maintaining standards-compliant SKOS data.

## Features

- **SKOS → Notion**: Convert SKOS Turtle files to Notion-compatible formats (Markdown, CSV, JSON)
- **Notion → SKOS**: Convert Notion markdown exports back to SKOS Turtle
- **Validation**: Comprehensive SKOS validation with error and warning detection
- **Hierarchy Preservation**: Maintains broader/narrower relationships and concept schemes
- **Metadata Support**: Preserves definitions, alternative labels, notations, and URIs
- **UUID Generation**: Automatically creates unique identifiers for new concepts

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

#### From Markdown:
1. Create a new Notion page
2. Copy the entire markdown content
3. Paste into Notion (Cmd/Ctrl+V)
4. Optional: Convert to toggle lists:
   - Select hierarchical sections
   - Press Cmd/Ctrl+Shift+7

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

## Example Workflows

### Workflow 1: Create New Vocabulary in Notion
1. Create structured markdown in Notion following the format above
2. Export from Notion as markdown
3. Convert to SKOS: `python3 skos-to-notion-converter.py to-skos export.md`
4. Use the generated .ttl file in your semantic web applications

### Workflow 2: Edit Existing SKOS Vocabulary
1. Convert SKOS to Notion: `python3 skos-to-notion-converter.py to-notion vocab.ttl --format markdown`
2. Import markdown to Notion
3. Edit in Notion (add concepts, reorganize hierarchy, update definitions)
4. Export from Notion
5. Convert back: `python3 skos-to-notion-converter.py to-skos edited_export.md`

### Workflow 3: Team Collaboration
1. Convert SKOS to CSV: `python3 skos-to-notion-converter.py to-notion vocab.ttl --format csv`
2. Import as Notion database
3. Share with team for collaborative editing
4. Add comments, filters, and views
5. Export and convert back to SKOS when ready

## Output Examples

### SKOS Input (Turtle):
```turtle
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix ex: <http://example.org/vocab#> .

ex:Animals a skos:ConceptScheme ;
    skos:prefLabel "Animal Classification" .

ex:Mammals a skos:Concept ;
    skos:prefLabel "Mammals" ;
    skos:definition "Warm-blooded vertebrates" ;
    skos:topConceptOf ex:Animals ;
    skos:inScheme ex:Animals .

ex:Dogs a skos:Concept ;
    skos:prefLabel "Dogs" ;
    skos:broader ex:Mammals ;
    skos:inScheme ex:Animals .
```

### Notion Markdown Output:
```markdown
## 📂 Concept Scheme: Animal Classification

### ▸ Mammals
_Definition:_ Warm-blooded vertebrates  
<sub>URI: http://example.org/vocab#Mammals</sub>

#### ▹ Dogs
<sub>URI: http://example.org/vocab#Dogs</sub>
```

## Tips and Best Practices

1. **Validation First**: Always run without `--skip-validation` first to catch issues
2. **Namespace Planning**: Choose meaningful namespaces before converting from Notion
3. **Definition Quality**: Add meaningful definitions in Notion to avoid "Lorem ipsum" placeholders
4. **URI Preservation**: Existing URIs in markdown are preserved during round-trip conversion
5. **Hierarchy Limits**: Keep hierarchies under 7 levels deep for better usability
6. **Regular Backups**: Keep copies of both SKOS and Notion versions

## Troubleshooting

### Common Issues:

**"Bad syntax" error when parsing Turtle**
- Check for missing periods at end of statements
- Verify all URIs are enclosed in angle brackets
- Ensure quotes are properly closed

**"Maximum recursion depth exceeded"**
- Your SKOS file has circular references
- Run validation to identify the cycle
- Fix the circular broader/narrower relationships

**Missing concepts after conversion**
- Check for duplicate URIs in source data
- Verify all concepts have required properties
- Review validation warnings

**UTF-8 encoding errors**
- Ensure your files are saved with UTF-8 encoding
- Use the `--encoding utf-8` flag if needed


## Acknowledgments

Built with [RDFLib](https://rdflib.readthedocs.io/) for RDF processing.
