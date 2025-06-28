#!/usr/bin/env python3
"""
SKOS RDF to Notion Converter (Refactored)
Converts SKOS vocabularies in Turtle format to CSV/Markdown for Notion import
Also converts Notion markdown back to SKOS Turtle format
"""

# Standard library imports
import argparse
import csv
import json
import os
import re
import sys
import uuid
from collections import defaultdict
from io import StringIO

# Third-party imports
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import SKOS, RDF, RDFS, DC, DCTERMS


class SKOSToNotionConverter:
    """Converter for SKOS RDF to Notion-compatible formats."""
    
    def __init__(self):
        self.graph = Graph()
        self.skos = Namespace("http://www.w3.org/2004/02/skos/core#")
        self.markdown_style = 'headings'  # Default style
        
    def load_turtle(self, file_path):
        """Load Turtle RDF file"""
        try:
            self.graph.parse(file_path, format='turtle')
            print(f"Loaded {len(self.graph)} triples from {file_path}")
        except FileNotFoundError:
            print(f"❌ Error: File not found: {file_path}")
            raise
        except PermissionError:
            print(f"❌ Error: Permission denied reading file: {file_path}")
            raise
        except (ValueError, TypeError, AttributeError) as e:
            self._handle_parse_error(file_path, e)
            raise
            
    def _handle_parse_error(self, file_path, error):
        """Handle parsing errors with detailed information."""
        error_type = type(error).__name__
        error_msg = str(error)
        
        print(f"❌ Error parsing Turtle file: {error_type}")
        print(f"   Details: {error_msg}")
        
        # Try to extract line number from error if available
        line_match = re.search(r'line (\d+)', error_msg)
        if line_match:
            line_num = line_match.group(1)
            print(f"   Error at line: {line_num}")
            self._show_error_context(file_path, int(line_num))
        
        print("\n📌 Common Turtle syntax issues:")
        print("   - Missing '.' at the end of statements")
        print("   - Missing ';' between properties of the same subject")
        print("   - Unclosed brackets or quotes")
        print("   - Invalid URIs (missing < > brackets)")
        print("   - Invalid escape sequences in strings")
        print("   - Malformed prefixes or namespaces")
        print("\n💡 Tip: Try validating your Turtle file at: http://ttl.summerofcode.be/")
        
    def _show_error_context(self, file_path, line_num):
        """Show context around error line."""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                line_idx = line_num - 1
                if 0 <= line_idx < len(lines):
                    print(f"   Line {line_num}: {lines[line_idx].strip()}")
                    if line_idx > 0:
                        print(f"   Line {line_num-1}: {lines[line_idx-1].strip()}")
                    if line_idx < len(lines) - 1:
                        print(f"   Line {line_num+1}: {lines[line_idx+1].strip()}")
        except (IOError, OSError):
            pass
        
    def get_label(self, uri):
        """Get preferred label for a concept"""
        labels = list(self.graph.objects(uri, SKOS.prefLabel))
        if labels:
            return str(labels[0])
        # Fallback to altLabel or literal
        labels = list(self.graph.objects(uri, SKOS.altLabel))
        if labels:
            return str(labels[0])
        # Fallback to rdfs:label
        labels = list(self.graph.objects(uri, RDFS.label))
        if labels:
            return str(labels[0])
        # Last resort: use local part of URI
        return str(uri).split('/')[-1].split('#')[-1]
    
    def get_definition(self, uri):
        """Get definition for a concept"""
        definitions = list(self.graph.objects(uri, SKOS.definition))
        if definitions:
            return str(definitions[0])
        # Fallback to scopeNote
        notes = list(self.graph.objects(uri, SKOS.scopeNote))
        if notes:
            return str(notes[0])
        return ""
    
    def get_alt_labels(self, uri):
        """Get alternative labels"""
        alt_labels = list(self.graph.objects(uri, SKOS.altLabel))
        return [str(label) for label in alt_labels]
    
    def get_notation(self, uri):
        """Get notation/code for a concept"""
        notations = list(self.graph.objects(uri, SKOS.notation))
        return str(notations[0]) if notations else ""
    
    def build_hierarchy(self):
        """Build hierarchical structure from SKOS broader/narrower relations"""
        hierarchy = defaultdict(list)
        all_concepts = set()
        top_concepts = set()
        concept_to_scheme = {}  # Track which scheme each concept belongs to
        
        # Find all concepts
        for s in self.graph.subjects(RDF.type, SKOS.Concept):
            all_concepts.add(s)
            
        # Find concept schemes and their top concepts
        schemes = self._build_concept_schemes(all_concepts, concept_to_scheme, top_concepts)
        
        # Build parent-child relationships
        self._build_parent_child_relationships(all_concepts, hierarchy)
        
        # Detect and fix circular references
        self._detect_circular_references(all_concepts, hierarchy)
        
        # Find orphan concepts
        orphans_by_scheme, orphans_no_scheme = self._find_orphan_concepts(
            all_concepts, top_concepts, hierarchy, concept_to_scheme
        )
        
        return schemes, hierarchy, top_concepts, orphans_by_scheme, orphans_no_scheme
    
    def _build_concept_schemes(self, all_concepts, concept_to_scheme, top_concepts):
        """Build concept schemes dictionary."""
        schemes = {}
        for scheme in self.graph.subjects(RDF.type, SKOS.ConceptScheme):
            scheme_label = self.get_label(scheme)
            schemes[scheme] = {
                'label': scheme_label,
                'top_concepts': set(),
                'definition': self.get_definition(scheme)  # Added scheme definition
            }
            
            # Get top concepts via hasTopConcept
            for top_concept in self.graph.objects(scheme, SKOS.hasTopConcept):
                schemes[scheme]['top_concepts'].add(top_concept)
                top_concepts.add(top_concept)
                concept_to_scheme[top_concept] = scheme
                
            # Get top concepts via topConceptOf
            for top_concept in self.graph.subjects(SKOS.topConceptOf, scheme):
                schemes[scheme]['top_concepts'].add(top_concept)
                top_concepts.add(top_concept)
                concept_to_scheme[top_concept] = scheme
                
            # Get concepts via inScheme
            for concept in self.graph.subjects(SKOS.inScheme, scheme):
                concept_to_scheme[concept] = scheme
                
        return schemes
    
    def _build_parent_child_relationships(self, all_concepts, hierarchy):
        """Build parent-child relationships ensuring each child appears only once."""
        children_assigned = set()
        
        for concept in all_concepts:
            # Get narrower concepts (children)
            for narrower in self.graph.objects(concept, SKOS.narrower):
                if narrower != concept and narrower not in children_assigned:
                    hierarchy[concept].append(narrower)
                    children_assigned.add(narrower)
                    
            # Get via broader relations (inverse) - only if not already assigned
            for child in self.graph.subjects(SKOS.broader, concept):
                if child != concept and child not in children_assigned and child not in hierarchy[concept]:
                    hierarchy[concept].append(child)
                    children_assigned.add(child)
    
    def _detect_circular_references(self, all_concepts, hierarchy):
        """Detect and remove circular references."""
        def has_circular_reference(node, visited, path):
            if node in path:
                return True
            if node in visited:
                return False
            visited.add(node)
            path.add(node)
            for child in hierarchy.get(node, []):
                if has_circular_reference(child, visited, path):
                    print(f"Warning: Circular reference detected involving {self.get_label(node)}")
                    # Remove the circular reference
                    hierarchy[node] = [c for c in hierarchy[node] if c not in path]
                    return False
            path.remove(node)
            return False
        
        # Check all nodes for circular references
        for concept in all_concepts:
            has_circular_reference(concept, set(), set())
    
    def _find_orphan_concepts(self, all_concepts, top_concepts, hierarchy, concept_to_scheme):
        """Find orphan concepts grouped by scheme."""
        # Find orphan concepts
        orphans = set()
        children_assigned = set()
        for parent in hierarchy:
            children_assigned.update(hierarchy[parent])
            
        for concept in all_concepts:
            has_broader = bool(list(self.graph.objects(concept, SKOS.broader)))
            is_top = concept in top_concepts
            is_child = concept in children_assigned
            
            if not has_broader and not is_top and not is_child:
                orphans.add(concept)
        
        # Group orphans by scheme
        orphans_by_scheme = defaultdict(set)
        orphans_no_scheme = set()
        
        for orphan in orphans:
            if orphan in concept_to_scheme:
                orphans_by_scheme[concept_to_scheme[orphan]].add(orphan)
            else:
                orphans_no_scheme.add(orphan)
                
        return orphans_by_scheme, orphans_no_scheme
    
    def validate_skos(self):
        """Validate SKOS data and report issues"""
        print("\n=== SKOS Validation Report ===\n")
        
        issues = []
        warnings = []
        
        # Track all concepts and schemes
        concepts = set(self.graph.subjects(RDF.type, SKOS.Concept))
        schemes = set(self.graph.subjects(RDF.type, SKOS.ConceptScheme))
        
        print(f"Found {len(concepts)} concepts and {len(schemes)} concept schemes\n")
        
        # Run validation checks
        self._check_duplicate_uris(concepts, schemes, issues)
        self._check_missing_labels(concepts, issues)
        self._check_circular_references(concepts, issues)
        self._check_concepts_without_schemes(concepts, warnings)
        self._check_duplicate_labels(concepts, warnings)
        self._check_polyhierarchy(concepts, warnings)
        self._check_orphan_concepts(concepts, schemes, warnings)
        self._check_hierarchy_depth(schemes, warnings)
        self._check_multiple_pref_labels(concepts, issues)
        
        # Print results
        self._print_validation_results(issues, warnings)
        
        return len(issues) == 0  # Return True if no critical issues
    
    def _check_duplicate_uris(self, concepts, schemes, issues):
        """Check for duplicate URIs."""
        print("Checking for duplicate URIs...")
        all_resources = list(concepts) + list(schemes)
        uri_counts = defaultdict(int)
        for resource in all_resources:
            uri_counts[str(resource)] += 1
        
        for uri, count in uri_counts.items():
            if count > 1:
                issues.append(f"Duplicate URI found {count} times: {uri}")
    
    def _check_missing_labels(self, concepts, issues):
        """Check for missing labels."""
        print("Checking for missing labels...")
        for concept in concepts:
            if not list(self.graph.objects(concept, SKOS.prefLabel)):
                if not list(self.graph.objects(concept, RDFS.label)):
                    issues.append(f"Concept {concept} has no prefLabel or rdfs:label")
    
    def _check_circular_references(self, concepts, issues):
        """Check for circular broader/narrower relationships."""
        print("Checking for circular broader/narrower relationships...")
        
        def find_circular_refs(start, current, path, visited_paths):
            if current in path:
                return path + [current]
            if (start, current) in visited_paths:
                return None
            visited_paths.add((start, current))
            
            path = path + [current]
            for broader in self.graph.objects(current, SKOS.broader):
                result = find_circular_refs(start, broader, path, visited_paths)
                if result:
                    return result
            return None
        
        circular_refs = set()
        for concept in concepts:
            visited_paths = set()
            circular_path = find_circular_refs(concept, concept, [], visited_paths)
            if circular_path and len(circular_path) > 2:
                # Convert to labels for readability
                path_labels = [self.get_label(c) for c in circular_path]
                circular_refs.add(" -> ".join(path_labels))
        
        for ref in circular_refs:
            issues.append(f"Circular reference detected: {ref}")
    
    def _check_concepts_without_schemes(self, concepts, warnings):
        """Check for concepts without concept schemes."""
        print("Checking for concepts without concept schemes...")
        orphan_concepts = []
        for concept in concepts:
            in_scheme = list(self.graph.objects(concept, SKOS.inScheme))
            if not in_scheme:
                orphan_concepts.append(self.get_label(concept))
        
        if orphan_concepts:
            warnings.append(f"{len(orphan_concepts)} concepts not associated with any concept scheme: "
                          f"{', '.join(orphan_concepts[:5])}{'...' if len(orphan_concepts) > 5 else ''}")
    
    def _check_duplicate_labels(self, concepts, warnings):
        """Check for duplicate preferred labels."""
        print("Checking for duplicate preferred labels...")
        label_map = defaultdict(list)
        for concept in concepts:
            labels = list(self.graph.objects(concept, SKOS.prefLabel))
            for label in labels:
                label_map[str(label)].append(concept)
        
        duplicate_labels = []
        for label, concepts_list in label_map.items():
            if len(concepts_list) > 1:
                concept_labels = [f"{self.get_label(c)} ({c})" for c in concepts_list[:3]]
                duplicate_labels.append(f"'{label}' used by: {', '.join(concept_labels)}"
                                      f"{'...' if len(concepts_list) > 3 else ''}")
        
        if duplicate_labels:
            warnings.append(f"{len(duplicate_labels)} duplicate preferred labels found "
                          "(valid but may cause confusion):")
            for dup in duplicate_labels[:5]:  # Show first 5
                warnings.append(f"  - {dup}")
            if len(duplicate_labels) > 5:
                warnings.append(f"  ... and {len(duplicate_labels) - 5} more")
    
    def _check_polyhierarchy(self, concepts, warnings):
        """Check for multiple broader concepts."""
        print("Checking for polyhierarchy...")
        polyhierarchy = []
        for concept in concepts:
            broaders = list(self.graph.objects(concept, SKOS.broader))
            if len(broaders) > 1:
                broader_labels = [self.get_label(b) for b in broaders]
                polyhierarchy.append(f"{self.get_label(concept)} has multiple broader concepts: "
                                   f"{', '.join(broader_labels)}")
        
        if polyhierarchy:
            warnings.append(f"{len(polyhierarchy)} concepts have multiple broader concepts "
                          "(polyhierarchy - valid but worth noting)")
            for p in polyhierarchy[:3]:  # Show first 3 examples
                warnings.append(f"  - {p}")
            if len(polyhierarchy) > 3:
                warnings.append(f"  ... and {len(polyhierarchy) - 3} more")
    
    def _check_orphan_concepts(self, concepts, schemes, warnings):
        """Check for orphan concepts."""
        print("Checking for orphan concepts...")
        top_concepts = set()
        for scheme in schemes:
            top_concepts.update(self.graph.objects(scheme, SKOS.hasTopConcept))
            top_concepts.update(self.graph.subjects(SKOS.topConceptOf, scheme))
        
        true_orphans = []
        for concept in concepts:
            broaders = list(self.graph.objects(concept, SKOS.broader))
            if not broaders and concept not in top_concepts:
                true_orphans.append(self.get_label(concept))
        
        if true_orphans:
            warnings.append(f"{len(true_orphans)} concepts have no broader concept "
                          "and are not marked as top concepts")
    
    def _check_hierarchy_depth(self, schemes, warnings):
        """Check for very deep hierarchies."""
        print("Checking hierarchy depth...")
        
        def get_depth(concept, visited=None):
            if visited is None:
                visited = set()
            if concept in visited:
                return 0
            visited.add(concept)
            
            narrowers = list(self.graph.objects(concept, SKOS.narrower))
            if not narrowers:
                return 1
            return 1 + max(get_depth(n, visited.copy()) for n in narrowers)
        
        # Get all top concepts
        top_concepts = set()
        for scheme in schemes:
            top_concepts.update(self.graph.objects(scheme, SKOS.hasTopConcept))
            top_concepts.update(self.graph.subjects(SKOS.topConceptOf, scheme))
        
        deep_hierarchies = []
        for concept in top_concepts:
            depth = get_depth(concept)
            if depth > 7:
                deep_hierarchies.append(f"{self.get_label(concept)}: {depth} levels")
        
        if deep_hierarchies:
            warnings.append("Very deep hierarchies detected:")
            for h in deep_hierarchies:
                warnings.append(f"  - {h}")
    
    def _check_multiple_pref_labels(self, concepts, issues):
        """Check for multiple preferred labels on single concept."""
        print("Checking for multiple labels on single concepts...")
        for concept in concepts:
            pref_labels = list(self.graph.objects(concept, SKOS.prefLabel))
            if len(pref_labels) > 1:
                issues.append(f"Concept {self.get_label(concept)} has {len(pref_labels)} "
                            "preferred labels (should have exactly one)")
    
    def _print_validation_results(self, issues, warnings):
        """Print validation results."""
        print("\n=== Validation Results ===\n")
        
        if not issues and not warnings:
            print("✓ No issues found! SKOS data appears to be well-formed.\n")
        else:
            if issues:
                print(f"ERRORS ({len(issues)}):")
                for issue in issues:
                    print(f"  ✗ {issue}")
                print()
            
            if warnings:
                print(f"WARNINGS ({len(warnings)}):")
                for warning in warnings:
                    print(f"  ⚠ {warning}")
                print()
    
    def to_notion_csv(self, output_file):
        """Convert to CSV format suitable for Notion import"""
        schemes, hierarchy, _, orphans_by_scheme, orphans_no_scheme = self.build_hierarchy()
        
        rows = []
        processed = set()
        
        def add_concept_row(concept, parent_label="", level=0, scheme_label=""):
            """Add a concept and its children to rows"""
            # Skip if already processed
            if concept in processed:
                return
                
            processed.add(concept)
            
            label = self.get_label(concept)
            definition = self.get_definition(concept)
            alt_labels = ", ".join(self.get_alt_labels(concept))
            notation = self.get_notation(concept)
            
            # Create indentation for visual hierarchy
            indented_label = "  " * level + label
            
            rows.append({
                'Title': indented_label,
                'Parent': parent_label,
                'Concept Scheme': scheme_label,
                'Definition': definition,
                'Alternative Labels': alt_labels,
                'Notation': notation,
                'URI': str(concept),
                'Level': level
            })
            
            # Add children in alphabetical order
            if concept in hierarchy:
                children = sorted(hierarchy[concept], key=self.get_label)
                for child in children:
                    add_concept_row(child, label, level + 1, scheme_label)
        
        # Process each concept scheme
        self._process_schemes_to_csv(schemes, rows, add_concept_row, orphans_by_scheme)
        
        # Add orphan concepts with no scheme
        self._process_orphans_to_csv(orphans_no_scheme, rows, add_concept_row)
        
        # Write CSV
        self._write_csv(output_file, rows)
        
        print(f"Created CSV with {len(rows)} entries")
        print(f"Processed {len(processed)} unique concepts")
    
    def _process_schemes_to_csv(self, schemes, rows, add_concept_row, orphans_by_scheme):
        """Process schemes for CSV output."""
        for scheme in sorted(schemes.keys(), key=lambda x: schemes[x]['label']):
            scheme_data = schemes[scheme]
            scheme_label = scheme_data['label']
            
            # Add scheme as top-level item
            rows.append({
                'Title': f"[SCHEME] {scheme_label}",
                'Parent': "",
                'Concept Scheme': scheme_label,
                'Definition': "",
                'Alternative Labels': "",
                'Notation': "",
                'URI': str(scheme),
                'Level': 0
            })
            
            # Add top concepts in alphabetical order
            sorted_top_concepts = sorted(scheme_data['top_concepts'], key=self.get_label)
            for top_concept in sorted_top_concepts:
                add_concept_row(top_concept, f"[SCHEME] {scheme_label}", 1, scheme_label)
                
            # Add orphans that belong to this scheme
            if scheme in orphans_by_scheme and orphans_by_scheme[scheme]:
                rows.append({
                    'Title': f"  [Other Concepts in {scheme_label}]",
                    'Parent': f"[SCHEME] {scheme_label}",
                    'Concept Scheme': scheme_label,
                    'Definition': "Concepts in this scheme without broader relations",
                    'Alternative Labels': "",
                    'Notation': "",
                    'URI': "",
                    'Level': 1
                })
                
                sorted_orphans = sorted(orphans_by_scheme[scheme], key=self.get_label)
                for orphan in sorted_orphans:
                    add_concept_row(orphan, f"[Other Concepts in {scheme_label}]", 2, scheme_label)
    
    def _process_orphans_to_csv(self, orphans_no_scheme, rows, add_concept_row):
        """Process orphan concepts for CSV output."""
        if orphans_no_scheme:
            rows.append({
                'Title': "[UNASSIGNED CONCEPTS]",
                'Parent': "",
                'Concept Scheme': "",
                'Definition': "Concepts not associated with any concept scheme",
                'Alternative Labels': "",
                'Notation': "",
                'URI': "",
                'Level': 0
            })
            
            sorted_orphans = sorted(orphans_no_scheme, key=self.get_label)
            for orphan in sorted_orphans:
                add_concept_row(orphan, "[UNASSIGNED CONCEPTS]", 1, "")
    
    def _write_csv(self, output_file, rows):
        """Write CSV file."""
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['Title', 'Parent', 'Concept Scheme', 'Definition', 
                         'Alternative Labels', 'Notation', 'URI', 'Level']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    
    def to_notion_markdown(self, output_file):
        """Convert to Markdown format with hierarchy for Notion import"""
        schemes, hierarchy, _, orphans_by_scheme, orphans_no_scheme = self.build_hierarchy()
        
        md_content = []
        md_content.append("# SKOS Vocabulary")
        md_content.append("")
        
        # Track all processed concepts to avoid duplicates
        processed = set()
        
        def add_concept_md(concept, level=1):
            """Add concept to markdown with proper heading level"""
            # Skip if already processed
            if concept in processed:
                return
                
            processed.add(concept)
            
            # Get concept metadata
            metadata = self._get_concept_metadata(concept)
            
            # Format concept based on level - extend to 6 levels
            self._format_concept_markdown(md_content, metadata, level)
            
            # Add children in alphabetical order
            if concept in hierarchy:
                children = sorted(hierarchy[concept], key=self.get_label)
                for child in children:
                    add_concept_md(child, level + 1)
        
        # Process concept schemes
        self._process_schemes_to_markdown(schemes, md_content, add_concept_md, orphans_by_scheme)
        
        # Add orphans with no scheme
        self._process_orphans_to_markdown(orphans_no_scheme, md_content, add_concept_md)
        
        # Write markdown file
        self._write_markdown(output_file, md_content)
        
        print(f"Created Markdown file: {output_file}")
        print(f"Processed {len(processed)} unique concepts")
    
    def _get_concept_metadata(self, concept):
        """Get all metadata for a concept."""
        return {
            'uri': concept,
            'label': self.get_label(concept),
            'definition': self.get_definition(concept),
            'alt_labels': self.get_alt_labels(concept),
            'notation': self.get_notation(concept)
        }
    
    def _format_concept_markdown(self, md_content, metadata, level):
        """Format concept for markdown output - improved for Notion."""
        label = metadata['label']
        
        # Use proper markdown headers up to 6 levels
        if level <= 6:
            md_content.append(f"{'#' * level} {label}")
        else:
            # For levels deeper than 6, use bold text with indentation
            indent = "  " * (level - 6)
            md_content.append(f"{indent}**{label}**")
        
        md_content.append("")
        
        # Add metadata without HTML tags
        self._add_concept_metadata_to_markdown(md_content, metadata)
    
    def _add_concept_metadata_to_markdown(self, md_content, metadata):
        """Add concept metadata to markdown without HTML tags."""
        if metadata['notation']:
            md_content.append(f"**Notation:** `{metadata['notation']}`")
            md_content.append("")
        
        if metadata['definition']:
            md_content.append(f"**Definition:** {metadata['definition']}")
            md_content.append("")
        
        if metadata['alt_labels']:
            md_content.append(f"**Alternative Labels:** {', '.join(metadata['alt_labels'])}")
            md_content.append("")
        
        # Add URI without HTML tags - use plain text
        md_content.append(f"URI: {metadata['uri']}")
        md_content.append("")
    
    def _process_schemes_to_markdown(self, schemes, md_content, add_concept_md, orphans_by_scheme):
        """Process schemes for markdown output."""
        for scheme in sorted(schemes.keys(), key=lambda x: schemes[x]['label']):
            scheme_data = schemes[scheme]
            scheme_label = scheme_data['label']
            
            # Add scheme as H2 (removed "Concept Scheme:" prefix)
            md_content.append(f"## {scheme_label}")
            md_content.append("")
            
            # Add scheme definition if it exists
            if scheme_data.get('definition'):
                md_content.append(f"**Definition:** {scheme_data['definition']}")
                md_content.append("")
            
            # Add scheme URI
            md_content.append(f"URI: {scheme}")
            md_content.append("")
            md_content.append("---")
            md_content.append("")
            
            # Add top concepts in alphabetical order - start at H3
            sorted_top_concepts = sorted(scheme_data['top_concepts'], key=self.get_label)
            for top_concept in sorted_top_concepts:
                add_concept_md(top_concept, 3)
                
            # Add orphans that belong to this scheme
            if scheme in orphans_by_scheme and orphans_by_scheme[scheme]:
                md_content.append(f"### Other Concepts in {scheme_label}")
                md_content.append("")
                md_content.append("*Concepts in this scheme without broader relations*")
                md_content.append("")
                
                sorted_orphans = sorted(orphans_by_scheme[scheme], key=self.get_label)
                for orphan in sorted_orphans:
                    add_concept_md(orphan, 4)
    
    def _process_orphans_to_markdown(self, orphans_no_scheme, md_content, add_concept_md):
        """Process orphan concepts for markdown output."""
        if orphans_no_scheme:
            md_content.append("## Unassigned Concepts")
            md_content.append("")
            md_content.append("*Concepts not associated with any concept scheme*")
            md_content.append("")
            
            sorted_orphans = sorted(orphans_no_scheme, key=self.get_label)
            for orphan in sorted_orphans:
                add_concept_md(orphan, 3)
    
    def _write_markdown(self, output_file, md_content):
        """Write markdown file with formatting instructions."""
        final_content = []
        
        # Add custom formatting instructions at the top
        final_content.append("<!--")
        final_content.append("NOTION IMPORT TIPS:")
        final_content.append("1. Use Cmd/Ctrl+Shift+V to paste and preserve formatting")
        final_content.append("2. Convert to toggle lists: highlight text and press Cmd/Ctrl+Shift+7")
        final_content.append("3. Use synced blocks for concepts that appear in multiple places")
        final_content.append("-->")
        final_content.append("")
        
        final_content.extend(md_content)
        
        # Write markdown
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(final_content))
    
    def to_notion_json(self, output_file):
        """Convert to JSON format that can be processed for Notion API"""
        schemes, hierarchy, _, orphans_by_scheme, orphans_no_scheme = self.build_hierarchy()
        
        notion_data = {
            "vocabulary": {
                "schemes": [],
                "concepts": []
            }
        }
        
        processed = set()
        
        def build_concept_dict(concept, parent_id=None):
            """Build concept dictionary"""
            if concept in processed:
                return None
                
            processed.add(concept)
            
            concept_id = str(concept).replace('/', '_').replace('#', '_')
            concept_dict = {
                "id": concept_id,
                "title": self.get_label(concept),
                "parent_id": parent_id,
                "definition": self.get_definition(concept),
                "alternative_labels": self.get_alt_labels(concept),
                "notation": self.get_notation(concept),
                "uri": str(concept),
                "children": []
            }
            
            # Add children in alphabetical order
            if concept in hierarchy:
                children = sorted(hierarchy[concept], key=self.get_label)
                for child in children:
                    child_dict = build_concept_dict(child, concept_id)
                    if child_dict:
                        concept_dict["children"].append(child_dict)
                        notion_data["vocabulary"]["concepts"].append(child_dict)
            
            return concept_dict
        
        # Process schemes
        self._process_schemes_to_json(schemes, notion_data, build_concept_dict, orphans_by_scheme)
        
        # Add orphans with no scheme
        self._process_orphans_to_json(orphans_no_scheme, notion_data, build_concept_dict)
        
        # Write JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(notion_data, f, indent=2, ensure_ascii=False)
            
        print(f"Created JSON file: {output_file}")
        print(f"Processed {len(processed)} unique concepts")
    
    def _process_schemes_to_json(self, schemes, notion_data, build_concept_dict, orphans_by_scheme):
        """Process schemes for JSON output."""
        for scheme in sorted(schemes.keys(), key=lambda x: schemes[x]['label']):
            scheme_data = schemes[scheme]
            scheme_id = str(scheme).replace('/', '_').replace('#', '_')
            scheme_dict = {
                "id": scheme_id,
                "title": scheme_data['label'],
                "uri": str(scheme),
                "definition": scheme_data.get('definition', ''),  # Include scheme definition
                "top_concepts": [],
                "other_concepts": []
            }
            
            # Add top concepts in alphabetical order
            sorted_top_concepts = sorted(scheme_data['top_concepts'], key=self.get_label)
            for top_concept in sorted_top_concepts:
                concept_dict = build_concept_dict(top_concept, scheme_id)
                if concept_dict:
                    scheme_dict["top_concepts"].append(concept_dict)
                    notion_data["vocabulary"]["concepts"].append(concept_dict)
                    
            # Add orphans that belong to this scheme
            if scheme in orphans_by_scheme and orphans_by_scheme[scheme]:
                sorted_orphans = sorted(orphans_by_scheme[scheme], key=self.get_label)
                for orphan in sorted_orphans:
                    concept_dict = build_concept_dict(orphan, scheme_id)
                    if concept_dict:
                        scheme_dict["other_concepts"].append(concept_dict)
                        notion_data["vocabulary"]["concepts"].append(concept_dict)
            
            notion_data["vocabulary"]["schemes"].append(scheme_dict)
    
    def _process_orphans_to_json(self, orphans_no_scheme, notion_data, build_concept_dict):
        """Process orphan concepts for JSON output."""
        if orphans_no_scheme:
            sorted_orphans = sorted(orphans_no_scheme, key=self.get_label)
            unassigned_concepts = []
            for orphan in sorted_orphans:
                concept_dict = build_concept_dict(orphan, None)
                if concept_dict:
                    unassigned_concepts.append(concept_dict)
                    notion_data["vocabulary"]["concepts"].append(concept_dict)
                    
            if unassigned_concepts:
                notion_data["vocabulary"]["unassigned_concepts"] = unassigned_concepts


class NotionToSKOSConverter:
    """Converter for Notion markdown to SKOS RDF format."""
    
    def __init__(self, namespace_uri="http://example.org/vocabulary#", prefix="ex"):
        self.namespace_uri = namespace_uri.rstrip('#/') + '#'
        self.prefix = prefix
        self.namespace = Namespace(self.namespace_uri)
        self.graph = Graph()
        self.graph.bind(prefix, self.namespace)
        self.graph.bind('skos', SKOS)
        self.graph.bind('rdf', RDF)
        self.graph.bind('rdfs', RDFS)
        self.existing_uris = {}  # Track label to URI mapping
        
    def parse_markdown(self, file_path):
        """Parse Notion markdown file and build SKOS graph"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"❌ Error: Markdown file not found: {file_path}")
            raise
        except PermissionError:
            print(f"❌ Error: Permission denied reading file: {file_path}")
            raise
        except UnicodeDecodeError as e:
            print(f"❌ Error: File encoding issue - {e}")
            print("   Try saving the file with UTF-8 encoding")
            raise
        except (IOError, OSError) as e:
            print(f"❌ Error reading markdown file: {type(e).__name__}: {e}")
            raise
        
        current_scheme = None
        current_parent_stack = []  # Stack to track hierarchy
        i = 0
        
        print(f"\nParsing markdown file: {file_path}")
        print(f"Using namespace: {self.namespace_uri}")
        print(f"Using prefix: {self.prefix}\n")
        
        try:
            while i < len(lines):
                i = self._process_line(lines, i, current_scheme, current_parent_stack)
                
        except (ValueError, TypeError, AttributeError) as e:
            print(f"\n❌ Error parsing markdown at line {i+1}: {type(e).__name__}")
            print(f"   Details: {e}")
            if i < len(lines):
                print(f"   Line {i+1}: {lines[i].strip()}")
            raise
        
        print(f"Parsed {len(self.graph)} triples")
        return self.graph
    
    def _process_line(self, lines, i, current_scheme, current_parent_stack):
        """Process a single line of markdown."""
        line = lines[i].strip()
        
        # Skip empty lines and comments
        if not line or line.startswith('<!--'):
            return i + 1
            
        # Parse headers
        header_match = re.match(r'^(#+)\s+(.+)$', line)
        if header_match:
            level = len(header_match.group(1))
            title = header_match.group(2)
            
            # Clean and process title
            title = self._clean_title(title)
            
            # Skip special sections
            if self._should_skip_section(title):
                return i + 1
            
            # Extract metadata
            metadata = self._extract_metadata(lines, i)
            
            # Process based on level
            if level == 1:
                self._process_concept_scheme(title, metadata, current_scheme, current_parent_stack)
            elif level >= 2 and current_scheme:
                self._process_concept(title, metadata, level, current_scheme, current_parent_stack)
            elif level >= 2 and not current_scheme:
                print(f"⚠️  Warning: Found concept '{title}' at line {i+1} without a concept scheme (H1)")
                print("   Skipping this concept...")
        
        return i + 1
    
    def _clean_title(self, title):
        """Clean title by removing visual indicators."""
        # Remove visual indicators
        title = re.sub(r'^[▸▹◦📂📁📄]\s*', '', title)
        return title
    
    def _should_skip_section(self, title):
        """Check if section should be skipped."""
        return (title.startswith('[') or 
                title.startswith('Other Concepts') or 
                title == 'Unassigned Concepts')
    
    def _extract_metadata(self, lines, start_index):
        """Extract metadata from lines following a header."""
        metadata = {
            'definition': None,
            'alt_labels': [],
            'notation': None,
            'existing_uri': None
        }
        
        j = start_index + 1
        while j < len(lines) and not lines[j].strip().startswith('#'):
            metadata_line = lines[j].strip()
            
            # Parse different metadata types
            if metadata_line.startswith('**Definition:**'):
                metadata['definition'] = metadata_line.split(':', 1)[1].strip()
            
            elif metadata_line.startswith('**Alternative Labels:**'):
                alt_text = metadata_line.split(':', 1)[1].strip()
                metadata['alt_labels'] = [label.strip() for label in alt_text.split(',') if label.strip()]
            
            elif metadata_line.startswith('**Notation:**'):
                metadata['notation'] = metadata_line.split(':', 1)[1].strip().strip('`')
            
            elif metadata_line.startswith('URI:'):
                uri_text = metadata_line.replace('URI:', '').strip()
                if uri_text and uri_text != 'None':
                    metadata['existing_uri'] = uri_text
            
            j += 1
            
        return metadata
    
    def _process_concept_scheme(self, title, metadata, current_scheme, current_parent_stack):
        """Process a concept scheme (H1)."""        
        # Create or reuse URI
        if metadata['existing_uri']:
            scheme_uri = URIRef(metadata['existing_uri'])
        else:
            scheme_uri = self.namespace[self._create_uri_fragment(title)]
        
        current_scheme = scheme_uri
        self.existing_uris[title] = scheme_uri
        
        # Add to graph
        self.graph.add((scheme_uri, RDF.type, SKOS.ConceptScheme))
        self.graph.add((scheme_uri, SKOS.prefLabel, Literal(title)))
        
        # Add scheme definition if present
        if metadata['definition']:
            self.graph.add((scheme_uri, SKOS.definition, Literal(metadata['definition'])))
        
        # Reset hierarchy tracking
        current_parent_stack.clear()
        current_parent_stack.append((1, scheme_uri, title))
        
        return current_scheme
    
    def _process_concept(self, title, metadata, level, current_scheme, current_parent_stack):
        """Process a concept (H2+)."""
        # Create or reuse concept URI
        if metadata['existing_uri']:
            concept_uri = URIRef(metadata['existing_uri'])
        else:
            # Generate new UUID-based URI
            concept_id = str(uuid.uuid4())
            concept_uri = self.namespace[concept_id]
        
        self.existing_uris[title] = concept_uri
        
        # Add basic concept info
        self.graph.add((concept_uri, RDF.type, SKOS.Concept))
        self.graph.add((concept_uri, SKOS.prefLabel, Literal(title)))
        self.graph.add((concept_uri, SKOS.inScheme, current_scheme))
        
        # Add definition (or placeholder)
        if metadata['definition']:
            self.graph.add((concept_uri, SKOS.definition, Literal(metadata['definition'])))
        else:
            self.graph.add((concept_uri, SKOS.definition, Literal("Lorem ipsum")))
        
        # Add alternative labels
        for alt_label in metadata['alt_labels']:
            self.graph.add((concept_uri, SKOS.altLabel, Literal(alt_label)))
        
        # Add notation if present
        if metadata['notation']:
            self.graph.add((concept_uri, SKOS.notation, Literal(metadata['notation'])))
        
        # Update parent stack to current level
        while current_parent_stack and current_parent_stack[-1][0] >= level:
            current_parent_stack.pop()
        
        # Establish relationships
        if level == 2:  # H2 = Top Concept
            self.graph.add((current_scheme, SKOS.hasTopConcept, concept_uri))
            self.graph.add((concept_uri, SKOS.topConceptOf, current_scheme))
        else:  # H3+ = narrower concepts
            if current_parent_stack:
                parent_uri = current_parent_stack[-1][1]
                self.graph.add((concept_uri, SKOS.broader, parent_uri))
                self.graph.add((parent_uri, SKOS.narrower, concept_uri))
        
        # Add to stack for potential children
        current_parent_stack.append((level, concept_uri, title))
    
    def _create_uri_fragment(self, label):
        """Create a URI fragment from a label"""
        # Simple conversion: lowercase, replace spaces with underscores
        fragment = re.sub(r'[^\w\s-]', '', label)
        fragment = re.sub(r'\s+', '_', fragment)
        return fragment.lower()
    
    def export_turtle(self, output_file):
        """Export the graph as Turtle"""
        turtle_content = self.graph.serialize(format='turtle')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(turtle_content)
        
        print(f"\nExported SKOS Turtle to: {output_file}")
        print(f"Total triples: {len(self.graph)}")
        
        # Count resources
        concepts = set(self.graph.subjects(RDF.type, SKOS.Concept))
        schemes = set(self.graph.subjects(RDF.type, SKOS.ConceptScheme))
        print(f"Concept Schemes: {len(schemes)}")
        print(f"Concepts: {len(concepts)}")
        
        # Show sample output
        print("\nFirst few lines of output:")
        print("=" * 50)
        lines = turtle_content.splitlines()
        for line in lines[:20]:
            print(line)
        if len(lines) > 20:
            print("...")
            print("=" * 50)


def main():
    """Main entry point for the SKOS-Notion converter."""
    print("Starting SKOS-Notion Converter...")
    
    try:
        parser = create_argument_parser()
        args = parser.parse_args()
        
        print("Parsing command line arguments...")
        print(f"Parsed args: {args}")
        
        if not args.command:
            print("No command specified. Showing help...")
            parser.print_help()
            return 0
        
        if args.command == 'to-notion':
            return handle_to_notion_conversion(args)
        elif args.command == 'to-skos':
            return handle_to_skos_conversion(args)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Conversion cancelled by user")
        return 1
    except (ValueError, TypeError, AttributeError, IOError, OSError) as e:
        print(f"\n❌ Unexpected error: {type(e).__name__}")
        print(f"   Details: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def create_argument_parser():
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description='Convert between SKOS RDF (Turtle) and Notion-compatible formats'
    )
    
    # Create subparsers for different operations
    subparsers = parser.add_subparsers(dest='command', help='Conversion direction')
    
    # SKOS to Notion
    create_skos_to_notion_parser(subparsers)
    
    # Notion to SKOS
    create_notion_to_skos_parser(subparsers)
    
    return parser


def create_skos_to_notion_parser(subparsers):
    """Create parser for SKOS to Notion conversion."""
    skos_parser = subparsers.add_parser('to-notion', help='Convert SKOS Turtle to Notion formats')
    skos_parser.add_argument('input_file', help='Input Turtle RDF file')
    skos_parser.add_argument('--format', choices=['csv', 'markdown', 'json', 'all'], 
                             default='csv', help='Output format (default: csv)')
    skos_parser.add_argument('--output', help='Output file name (without extension)')
    skos_parser.add_argument('--skip-validation', action='store_true', 
                             help='Skip SKOS validation checks')
    skos_parser.add_argument('--force', action='store_true',
                             help='Continue conversion even if validation finds errors')
    skos_parser.add_argument('--markdown-style', choices=['headings', 'bullets', 'mixed'], 
                             default='headings', help='Markdown formatting style (default: headings)')


def create_notion_to_skos_parser(subparsers):
    """Create parser for Notion to SKOS conversion."""
    notion_parser = subparsers.add_parser('to-skos', help='Convert Notion markdown to SKOS Turtle')
    notion_parser.add_argument('input_file', help='Input Notion markdown file')
    notion_parser.add_argument('--output', help='Output file name (default: input_skos.ttl)')
    notion_parser.add_argument('--namespace', default='http://example.org/vocabulary#',
                               help='Namespace URI for new concepts (default: http://example.org/vocabulary#)')
    notion_parser.add_argument('--prefix', default='ex',
                               help='Namespace prefix (default: ex)')


def handle_to_notion_conversion(args):
    """Handle SKOS to Notion conversion."""
    print("\nProcessing SKOS to Notion conversion...")
    print(f"Input file: {args.input_file}")
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"❌ Error: Input file not found: {args.input_file}")
        print(f"   Current directory: {os.getcwd()}")
        return 1
    
    print(f"Input file exists: {os.path.abspath(args.input_file)}")
    
    # Determine output base name
    if args.output:
        base_output = args.output
    else:
        base_output = args.input_file.rsplit('.', 1)[0] + '_notion'
    
    print(f"Output base name: {base_output}")
    
    # Create converter and load file
    converter = SKOSToNotionConverter()
    
    try:
        print("Loading Turtle file...")
        converter.load_turtle(args.input_file)
    except (ValueError, TypeError, AttributeError, IOError, OSError):
        print("\n❌ Failed to load Turtle file")
        return 1
    
    # Store markdown style preference
    if hasattr(args, 'markdown_style'):
        converter.markdown_style = args.markdown_style
    
    # Run validation unless skipped
    if not args.skip_validation:
        is_valid = converter.validate_skos()
        
        if not is_valid and not args.force:
            print("\n❌ Validation found critical errors. Conversion aborted.")
            print("   Use --force to convert anyway, or fix the issues and try again.")
            print("   Use --skip-validation to skip validation entirely.\n")
            return 1
        elif not is_valid and args.force:
            print("\n⚠️  Continuing with conversion despite errors...\n")
    
    # Perform conversion
    try:
        if args.format in ['csv', 'all']:
            converter.to_notion_csv(f"{base_output}.csv")
            
        if args.format in ['markdown', 'all']:
            converter.to_notion_markdown(f"{base_output}.md")
            
        if args.format in ['json', 'all']:
            converter.to_notion_json(f"{base_output}.json")
    except (ValueError, TypeError, AttributeError, IOError, OSError) as e:
        print(f"\n❌ Error during conversion: {type(e).__name__}")
        print(f"   Details: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print_notion_import_instructions()
    return 0


def handle_to_skos_conversion(args):
    """Handle Notion to SKOS conversion."""
    # Determine output file
    if args.output:
        output_file = args.output
    else:
        output_file = args.input_file.rsplit('.', 1)[0] + '_skos.ttl'
    
    # Configure namespace
    namespace, prefix = configure_namespace(args)
    
    # Create converter and parse
    converter = NotionToSKOSConverter(namespace_uri=namespace, prefix=prefix)
    
    try:
        converter.parse_markdown(args.input_file)
    except (ValueError, TypeError, AttributeError, IOError, OSError):
        print("\n❌ Failed to parse markdown file")
        return 1
        
    try:
        converter.export_turtle(output_file)
    except (IOError, OSError) as e:
        print(f"\n❌ Error exporting Turtle: {type(e).__name__}")
        print(f"   Details: {e}")
        return 1
    
    print_skos_conversion_summary()
    return 0


def configure_namespace(args):
    """Configure namespace for SKOS conversion."""
    namespace = args.namespace
    prefix = args.prefix
    
    # Prompt for namespace if using default
    if namespace == 'http://example.org/vocabulary#':
        print("\n" + "="*60)
        print("NAMESPACE CONFIGURATION")
        print("="*60)
        print(f"Current namespace: {namespace}")
        print(f"Current prefix: {prefix}")
        print("\nPress Enter to use these defaults, or type new values:")
        
        new_namespace = input("Namespace URI [http://example.org/vocabulary#]: ").strip()
        if new_namespace:
            namespace = new_namespace
            
        new_prefix = input("Namespace prefix [ex]: ").strip()
        if new_prefix:
            prefix = new_prefix
            
    return namespace, prefix


def print_notion_import_instructions():
    """Print instructions for importing into Notion."""
    print("\nConversion complete!")
    print("\nNotion Import Instructions:")
    print("1. For CSV: Import into Notion as a table database")
    print("   - Use 'Title' as the page title")
    print("   - 'Parent' column creates hierarchy")
    print("   - Filter/group by 'Concept Scheme' or 'Level'")
    print("2. For Markdown: Copy/paste into Notion page")
    print("   - Hierarchy preserved as nested headings")
    print("   - Use Cmd/Ctrl+Shift+7 to convert to toggle lists")
    print("3. For JSON: Use with Notion API for programmatic import")


def print_skos_conversion_summary():
    """Print summary of SKOS conversion rules."""
    print("\n✅ Notion to SKOS conversion complete!")
    print("\nConversion rules applied:")
    print("- H1 headers → SKOS Concept Schemes")
    print("- H2 headers → Top Concepts")
    print("- H3+ headers → Narrower concepts with broader relationships")
    print("- All concepts have skos:inScheme relationship")
    print("- New concepts assigned UUID-based URIs")
    print("- Missing definitions replaced with 'Lorem ipsum'")


if __name__ == "__main__":
    print(f"Python version: {sys.version}")
    print(f"Command line arguments: {sys.argv}")
    
    EXIT_CODE = main()
    sys.exit(EXIT_CODE if EXIT_CODE else 0)