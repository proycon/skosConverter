#!/usr/bin/env python3
"""
SKOS RDF to Notion Converter (Optimized)
Converts SKOS vocabularies in Turtle format to CSV/Markdown for Notion import
Also converts Notion markdown back to SKOS Turtle format

Features:
- Bidirectional conversion
- Language tag support
- Batch processing
- Memory optimization
- Comprehensive validation
- pylint-compliant code
"""

# Standard library imports
import argparse
import csv
import json
import logging
import os
import re
import sys
import uuid
from collections import defaultdict
from functools import lru_cache
from io import StringIO
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Union, Any

# Third-party imports
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import SKOS, RDF, RDFS, DC, DCTERMS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class ConverterConfig:
    """Configuration class for converter settings."""

    def __init__(self):
        self.namespace_uri = "http://example.org/vocabulary#"
        self.prefix = "ex"
        self.markdown_style = "headings"
        self.validation_level = "strict"
        self.max_hierarchy_depth = 10
        self.preferred_language = None
        self.fallback_languages = ["en", ""]
        self.batch_size = 100
        self.memory_limit_mb = 512

    def set_language_preferences(self, preferred: Optional[str],
                                 fallbacks: Optional[List[str]]):
        """Set language preferences for label extraction."""
        if preferred:
            self.preferred_language = preferred
        if fallbacks:
            self.fallback_languages = fallbacks


class URIManager:
    """Manages URI creation and caching for improved performance."""

    def __init__(self, namespace_uri: str):
        self.namespace_uri = namespace_uri.rstrip('#/') + '#'
        self.namespace = Namespace(self.namespace_uri)
        self.uri_cache: Dict[str, URIRef] = {}
        self.label_to_uri: Dict[str, URIRef] = {}

    @lru_cache(maxsize=1000)
    def create_uri_fragment(self, label: str) -> str:
        """Create a URI fragment from a label with caching."""
        fragment = re.sub(r'[^\w\s-]', '', label)
        fragment = re.sub(r'\s+', '_', fragment)
        return fragment.lower()

    def get_or_create_uri(self, label: str, existing_uri: str = None) -> URIRef:
        """Get existing URI or create new one."""
        if existing_uri:
            uri = URIRef(existing_uri)
            self.label_to_uri[label] = uri
            return uri

        if label in self.label_to_uri:
            return self.label_to_uri[label]

        # Generate new UUID-based URI for uniqueness
        concept_id = str(uuid.uuid4())
        uri = self.namespace[concept_id]
        self.label_to_uri[label] = uri
        return uri


class SKOSValidator:
    """SKOS validation logic separated from main converter."""

    def __init__(self, graph: Graph):
        self.graph = graph
        self.issues: List[str] = []
        self.warnings: List[str] = []

    def validate_all(self) -> bool:
        """Run all validation checks."""
        logger.info("Starting SKOS validation...")

        concepts = set(self.graph.subjects(RDF.type, SKOS.Concept))
        schemes = set(self.graph.subjects(RDF.type, SKOS.ConceptScheme))

        logger.info("Found %d concepts and %d concept schemes",
                    len(concepts), len(schemes))

        self._check_duplicate_uris(concepts, schemes)
        self._check_missing_labels(concepts)
        self._check_circular_references(concepts)
        self._check_multiple_pref_labels(concepts)
        self._check_concepts_without_schemes(concepts)
        self._check_duplicate_labels(concepts)
        self._check_polyhierarchy(concepts)
        self._check_orphan_concepts(concepts, schemes)
        self._check_hierarchy_depth(schemes)

        self._print_validation_results()
        return len(self.issues) == 0

    def _check_duplicate_uris(self, concepts: Set[URIRef],
                              schemes: Set[URIRef]):
        """Check for duplicate URIs."""
        all_resources = list(concepts) + list(schemes)
        uri_counts = defaultdict(int)
        for resource in all_resources:
            uri_counts[str(resource)] += 1

        for uri, count in uri_counts.items():
            if count > 1:
                self.issues.append(f"Duplicate URI found {count} times: {uri}")

    def _check_missing_labels(self, concepts: Set[URIRef]):
        """Check for missing labels."""
        for concept in concepts:
            if not list(self.graph.objects(concept, SKOS.prefLabel)):
                if not list(self.graph.objects(concept, RDFS.label)):
                    self.issues.append(
                        f"Concept {concept} has no prefLabel or rdfs:label"
                    )

    def _check_circular_references(self, concepts: Set[URIRef]):
        """Check for circular broader/narrower relationships."""
        def find_circular_refs(start, current, path, visited_paths):
            if current in path:
                return path + [current]
            if (start, current) in visited_paths:
                return None
            visited_paths.add((start, current))

            path = path + [current]
            for broader in self.graph.objects(current, SKOS.broader):
                result = find_circular_refs(start, broader, path,
                                            visited_paths)
                if result:
                    return result
            return None

        circular_refs = set()
        for concept in concepts:
            visited_paths = set()
            circular_path = find_circular_refs(concept, concept, [],
                                               visited_paths)
            if circular_path and len(circular_path) > 2:
                path_labels = [self._get_simple_label(c) for c in circular_path]
                circular_refs.add(" -> ".join(path_labels))

        for ref in circular_refs:
            self.issues.append(f"Circular reference detected: {ref}")

    def _check_multiple_pref_labels(self, concepts: Set[URIRef]):
        """Check for multiple preferred labels on single concept."""
        for concept in concepts:
            pref_labels = list(self.graph.objects(concept, SKOS.prefLabel))
            if len(pref_labels) > 1:
                self.issues.append(
                    f"Concept {self._get_simple_label(concept)} has "
                    f"{len(pref_labels)} preferred labels "
                    "(should have exactly one per language)"
                )

    def _check_concepts_without_schemes(self, concepts: Set[URIRef]):
        """Check for concepts without concept schemes."""
        orphan_concepts = []
        for concept in concepts:
            in_scheme = list(self.graph.objects(concept, SKOS.inScheme))
            if not in_scheme:
                orphan_concepts.append(self._get_simple_label(concept))

        if orphan_concepts:
            self.warnings.append(
                f"{len(orphan_concepts)} concepts not associated with "
                f"any concept scheme: "
                f"{', '.join(orphan_concepts[:5])}"
                f"{'...' if len(orphan_concepts) > 5 else ''}"
            )

    def _check_duplicate_labels(self, concepts: Set[URIRef]):
        """Check for duplicate preferred labels."""
        label_map = defaultdict(list)
        for concept in concepts:
            labels = list(self.graph.objects(concept, SKOS.prefLabel))
            for label in labels:
                label_map[str(label)].append(concept)

        duplicate_labels = []
        for label, concepts_list in label_map.items():
            if len(concepts_list) > 1:
                concept_labels = [f"{self._get_simple_label(c)} ({c})"
                                  for c in concepts_list[:3]]
                duplicate_labels.append(
                    f"'{label}' used by: {', '.join(concept_labels)}"
                    f"{'...' if len(concepts_list) > 3 else ''}"
                )

        if duplicate_labels:
            self.warnings.append(
                f"{len(duplicate_labels)} duplicate preferred labels found "
                "(valid but may cause confusion):"
            )
            for dup in duplicate_labels[:5]:
                self.warnings.append(f"  - {dup}")
            if len(duplicate_labels) > 5:
                self.warnings.append(f"  ... and {len(duplicate_labels) - 5} more")

    def _check_polyhierarchy(self, concepts: Set[URIRef]):
        """Check for multiple broader concepts."""
        polyhierarchy = []
        for concept in concepts:
            broaders = list(self.graph.objects(concept, SKOS.broader))
            if len(broaders) > 1:
                broader_labels = [self._get_simple_label(b) for b in broaders]
                polyhierarchy.append(
                    f"{self._get_simple_label(concept)} has multiple "
                    f"broader concepts: {', '.join(broader_labels)}"
                )

        if polyhierarchy:
            self.warnings.append(
                f"{len(polyhierarchy)} concepts have multiple broader "
                "concepts (polyhierarchy - valid but worth noting)"
            )
            for p in polyhierarchy[:3]:
                self.warnings.append(f"  - {p}")
            if len(polyhierarchy) > 3:
                self.warnings.append(f"  ... and {len(polyhierarchy) - 3} more")

    def _check_orphan_concepts(self, concepts: Set[URIRef],
                               schemes: Set[URIRef]):
        """Check for orphan concepts."""
        top_concepts = set()
        for scheme in schemes:
            top_concepts.update(self.graph.objects(scheme, SKOS.hasTopConcept))
            top_concepts.update(self.graph.subjects(SKOS.topConceptOf, scheme))

        true_orphans = []
        for concept in concepts:
            broaders = list(self.graph.objects(concept, SKOS.broader))
            if not broaders and concept not in top_concepts:
                true_orphans.append(self._get_simple_label(concept))

        if true_orphans:
            self.warnings.append(
                f"{len(true_orphans)} concepts have no broader concept "
                "and are not marked as top concepts"
            )

    def _check_hierarchy_depth(self, schemes: Set[URIRef]):
        """Check for very deep hierarchies."""
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

        top_concepts = set()
        for scheme in schemes:
            top_concepts.update(self.graph.objects(scheme, SKOS.hasTopConcept))
            top_concepts.update(self.graph.subjects(SKOS.topConceptOf, scheme))

        deep_hierarchies = []
        for concept in top_concepts:
            depth = get_depth(concept)
            if depth > 7:
                deep_hierarchies.append(
                    f"{self._get_simple_label(concept)}: {depth} levels"
                )

        if deep_hierarchies:
            self.warnings.append("Very deep hierarchies detected:")
            for h in deep_hierarchies:
                self.warnings.append(f"  - {h}")

    def _get_simple_label(self, uri: URIRef) -> str:
        """Get simple label for URI."""
        labels = list(self.graph.objects(uri, SKOS.prefLabel))
        if labels:
            return str(labels[0])
        return str(uri).split('/')[-1].split('#')[-1]

    def _print_validation_results(self):
        """Print validation results."""
        logger.info("=== Validation Results ===")

        if not self.issues and not self.warnings:
            logger.info("✓ No issues found! SKOS data appears to be well-formed.")
        else:
            if self.issues:
                logger.error("ERRORS (%d):", len(self.issues))
                for issue in self.issues:
                    logger.error("  ✗ %s", issue)

            if self.warnings:
                logger.warning("WARNINGS (%d):", len(self.warnings))
                for warning in self.warnings:
                    logger.warning("  ⚠ %s", warning)


class LanguageHelper:
    """Helper for handling multilingual SKOS labels."""

    def __init__(self, config: ConverterConfig):
        self.config = config

    def get_best_label(self, graph: Graph, uri: URIRef,
                       property_uri: URIRef = SKOS.prefLabel) -> str:
        """Get the best label based on language preferences."""
        labels = list(graph.objects(uri, property_uri))

        if not labels:
            # Try fallback properties
            for fallback_prop in [SKOS.altLabel, RDFS.label]:
                labels = list(graph.objects(uri, fallback_prop))
                if labels:
                    break

        if not labels:
            # Last resort: use local part of URI
            return str(uri).split('/')[-1].split('#')[-1]

        # If preferred language is set, look for it first
        if self.config.preferred_language:
            for label in labels:
                if (hasattr(label, 'language') and
                        label.language == self.config.preferred_language):
                    return str(label)

        # Try fallback languages
        for fallback_lang in self.config.fallback_languages:
            for label in labels:
                label_lang = getattr(label, 'language', '')
                if label_lang == fallback_lang:
                    return str(label)

        # Return first available label
        return str(labels[0])

    def get_all_labels_by_language(self, graph: Graph, uri: URIRef,
                                   property_uri: URIRef = SKOS.prefLabel
                                   ) -> Dict[str, List[str]]:
        """Get all labels organized by language."""
        labels_by_lang = defaultdict(list)
        labels = list(graph.objects(uri, property_uri))

        for label in labels:
            lang = getattr(label, 'language', '') or 'no-lang'
            labels_by_lang[lang].append(str(label))

        return dict(labels_by_lang)


class BatchProcessor:
    """Handles batch processing of multiple files."""

    def __init__(self, config: ConverterConfig):
        self.config = config

    def process_directory(self, input_dir: str, output_dir: str,
                          format_type: str, operation: str):
        """Process all files in a directory."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if operation == 'to-notion':
            pattern = "*.ttl"
        else:
            pattern = "*.md"

        files = list(input_path.glob(pattern))
        logger.info("Found %d files to process", len(files))

        for i, file_path in enumerate(files, 1):
            logger.info("Processing file %d/%d: %s", i, len(files), file_path.name)

            try:
                if operation == 'to-notion':
                    self._process_skos_file(file_path, output_path, format_type)
                else:
                    self._process_notion_file(file_path, output_path)

            except Exception as e:
                logger.error("Error processing %s: %s", file_path.name, e)
                continue

        logger.info("Batch processing complete")

    def _process_skos_file(self, file_path: Path, output_dir: Path,
                           format_type: str):
        """Process a single SKOS file."""
        converter = SKOSToNotionConverter(self.config)
        converter.load_turtle(str(file_path))

        base_name = file_path.stem
        if format_type in ['csv', 'all']:
            output_file = output_dir / f"{base_name}.csv"
            converter.to_notion_csv(str(output_file))

        if format_type in ['markdown', 'all']:
            output_file = output_dir / f"{base_name}.md"
            converter.to_notion_markdown(str(output_file))

        if format_type in ['json', 'all']:
            output_file = output_dir / f"{base_name}.json"
            converter.to_notion_json(str(output_file))

    def _process_notion_file(self, file_path: Path, output_dir: Path):
        """Process a single Notion markdown file."""
        converter = NotionToSKOSConverter(
            namespace_uri=self.config.namespace_uri,
            prefix=self.config.prefix
        )
        converter.parse_markdown(str(file_path))

        base_name = file_path.stem
        output_file = output_dir / f"{base_name}_skos.ttl"
        converter.export_turtle(str(output_file))


class SKOSToNotionConverter:
    """Optimized converter for SKOS RDF to Notion-compatible formats."""

    def __init__(self, config: ConverterConfig):
        self.config = config
        self.graph = Graph()
        self.skos = Namespace("http://www.w3.org/2004/02/skos/core#")
        self.language_helper = LanguageHelper(config)

    def load_turtle(self, file_path: str):
        """Load Turtle RDF file with error handling."""
        try:
            self.graph.parse(file_path, format='turtle')
            logger.info("Loaded %d triples from %s", len(self.graph), file_path)
        except FileNotFoundError:
            logger.error("File not found: %s", file_path)
            raise
        except PermissionError:
            logger.error("Permission denied reading file: %s", file_path)
            raise
        except Exception as e:
            self._handle_parse_error(file_path, e)
            raise

    def _handle_parse_error(self, file_path: str, error: Exception):
        """Handle parsing errors with detailed information."""
        error_type = type(error).__name__
        error_msg = str(error)

        logger.error("Error parsing Turtle file: %s", error_type)
        logger.error("Details: %s", error_msg)

        # Try to extract line number from error if available
        line_match = re.search(r'line (\d+)', error_msg)
        if line_match:
            line_num = line_match.group(1)
            logger.error("Error at line: %s", line_num)
            self._show_error_context(file_path, int(line_num))

        logger.info("Common Turtle syntax issues:")
        logger.info("- Missing '.' at the end of statements")
        logger.info("- Missing ';' between properties of the same subject")
        logger.info("- Unclosed brackets or quotes")
        logger.info("- Invalid URIs (missing < > brackets)")
        logger.info("Tip: Try validating at: http://ttl.summerofcode.be/")

    def _show_error_context(self, file_path: str, line_num: int):
        """Show context around error line."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                line_idx = line_num - 1
                if 0 <= line_idx < len(lines):
                    logger.error("Line %d: %s", line_num, lines[line_idx].strip())
                    if line_idx > 0:
                        logger.error("Line %d: %s", line_num-1,
                                     lines[line_idx-1].strip())
                    if line_idx < len(lines) - 1:
                        logger.error("Line %d: %s", line_num+1,
                                     lines[line_idx+1].strip())
        except (IOError, OSError):
            pass

    def get_label(self, uri: URIRef) -> str:
        """Get preferred label for a concept using language preferences."""
        return self.language_helper.get_best_label(self.graph, uri,
                                                   SKOS.prefLabel)

    def get_definition(self, uri: URIRef) -> str:
        """Get definition for a concept."""
        return self.language_helper.get_best_label(self.graph, uri,
                                                   SKOS.definition) or ""

    def get_alt_labels(self, uri: URIRef) -> List[str]:
        """Get alternative labels."""
        alt_labels = list(self.graph.objects(uri, SKOS.altLabel))
        return [str(label) for label in alt_labels]

    def get_notation(self, uri: URIRef) -> str:
        """Get notation/code for a concept."""
        notations = list(self.graph.objects(uri, SKOS.notation))
        return str(notations[0]) if notations else ""

    def validate_skos(self) -> bool:
        """Validate SKOS data using the validator."""
        validator = SKOSValidator(self.graph)
        return validator.validate_all()

    def build_hierarchy(self) -> Tuple[Dict, Dict, Set, Dict, Set]:
        """Build hierarchical structure from SKOS broader/narrower relations."""
        hierarchy = defaultdict(list)
        all_concepts = set(self.graph.subjects(RDF.type, SKOS.Concept))
        top_concepts = set()
        concept_to_scheme = {}

        # Build concept schemes
        schemes = self._build_concept_schemes(all_concepts, concept_to_scheme,
                                              top_concepts)

        # Build parent-child relationships
        self._build_parent_child_relationships(all_concepts, hierarchy)

        # Find orphan concepts
        orphans_by_scheme, orphans_no_scheme = self._find_orphan_concepts(
            all_concepts, top_concepts, hierarchy, concept_to_scheme
        )

        return schemes, hierarchy, top_concepts, orphans_by_scheme, orphans_no_scheme

    def _build_concept_schemes(self, all_concepts: Set[URIRef],
                               concept_to_scheme: Dict[URIRef, URIRef],
                               top_concepts: Set[URIRef]) -> Dict:
        """Build concept schemes dictionary."""
        schemes = {}
        for scheme in self.graph.subjects(RDF.type, SKOS.ConceptScheme):
            scheme_label = self.get_label(scheme)
            schemes[scheme] = {
                'label': scheme_label,
                'top_concepts': set()
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

    def _build_parent_child_relationships(self, all_concepts: Set[URIRef],
                                          hierarchy: Dict):
        """Build parent-child relationships ensuring each child appears once."""
        children_assigned = set()

        for concept in all_concepts:
            # Get narrower concepts (children)
            for narrower in self.graph.objects(concept, SKOS.narrower):
                if narrower != concept and narrower not in children_assigned:
                    hierarchy[concept].append(narrower)
                    children_assigned.add(narrower)

            # Get via broader relations (inverse)
            for child in self.graph.subjects(SKOS.broader, concept):
                if (child != concept and child not in children_assigned and
                        child not in hierarchy[concept]):
                    hierarchy[concept].append(child)
                    children_assigned.add(child)

    def _find_orphan_concepts(self, all_concepts: Set[URIRef],
                              top_concepts: Set[URIRef], hierarchy: Dict,
                              concept_to_scheme: Dict) -> Tuple[Dict, Set]:
        """Find orphan concepts grouped by scheme."""
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

    def to_notion_csv(self, output_file: str):
        """Convert to CSV format suitable for Notion import."""
        schemes, hierarchy, _, orphans_by_scheme, orphans_no_scheme = \
            self.build_hierarchy()

        rows = []
        processed = set()

        def add_concept_row(concept, parent_label="", level=0, scheme_label=""):
            """Add a concept and its children to rows."""
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
        self._process_schemes_to_csv(schemes, rows, add_concept_row,
                                     orphans_by_scheme)

        # Add orphan concepts with no scheme
        self._process_orphans_to_csv(orphans_no_scheme, rows, add_concept_row)

        # Write CSV
        self._write_csv(output_file, rows)

        logger.info("Created CSV with %d entries", len(rows))
        logger.info("Processed %d unique concepts", len(processed))

    def _process_schemes_to_csv(self, schemes: Dict, rows: List,
                                add_concept_row, orphans_by_scheme: Dict):
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
            sorted_top_concepts = sorted(scheme_data['top_concepts'],
                                         key=self.get_label)
            for top_concept in sorted_top_concepts:
                add_concept_row(top_concept, f"[SCHEME] {scheme_label}", 1,
                                scheme_label)

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

                sorted_orphans = sorted(orphans_by_scheme[scheme],
                                        key=self.get_label)
                for orphan in sorted_orphans:
                    add_concept_row(orphan, f"[Other Concepts in {scheme_label}]",
                                    2, scheme_label)

    def _process_orphans_to_csv(self, orphans_no_scheme: Set, rows: List,
                                add_concept_row):
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

    def _write_csv(self, output_file: str, rows: List):
        """Write CSV file."""
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['Title', 'Parent', 'Concept Scheme', 'Definition',
                          'Alternative Labels', 'Notation', 'URI', 'Level']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def to_notion_markdown(self, output_file: str):
        """Convert to Markdown format with hierarchy for Notion import."""
        schemes, hierarchy, _, orphans_by_scheme, orphans_no_scheme = \
            self.build_hierarchy()

        md_content = []
        md_content.append("# SKOS Vocabulary\n")

        processed = set()

        def add_concept_md(concept, level=1, use_bullets=False):
            """Add concept to markdown with proper heading level."""
            if concept in processed:
                return

            processed.add(concept)

            # Get concept metadata
            metadata = self._get_concept_metadata(concept)

            # Format concept based on style
            self._format_concept_markdown(md_content, metadata, level,
                                          use_bullets)

            # Add children in alphabetical order
            if concept in hierarchy:
                children = sorted(hierarchy[concept], key=self.get_label)
                if children and not use_bullets:
                    md_content.append("")
                for child in children:
                    add_concept_md(child, level + 1, use_bullets)

        # Add table of contents
        self._add_table_of_contents(md_content, schemes, orphans_no_scheme)

        # Process concept schemes
        self._process_schemes_to_markdown(schemes, md_content, add_concept_md,
                                          orphans_by_scheme)

        # Add orphans with no scheme
        self._process_orphans_to_markdown(orphans_no_scheme, md_content,
                                          add_concept_md)

        # Write markdown file
        self._write_markdown(output_file, md_content)

        logger.info("Created Markdown file: %s", output_file)
        logger.info("Processed %d unique concepts", len(processed))

    def _get_concept_metadata(self, concept: URIRef) -> Dict[str, Any]:
        """Get all metadata for a concept."""
        return {
            'uri': concept,
            'label': self.get_label(concept),
            'definition': self.get_definition(concept),
            'alt_labels': self.get_alt_labels(concept),
            'notation': self.get_notation(concept)
        }

    def _format_concept_markdown(self, md_content: List, metadata: Dict,
                                 level: int, use_bullets: bool):
        """Format concept for markdown output."""
        label = metadata['label']

        if use_bullets:
            # Bullet list style with indentation
            indent = "  " * (level - 3)
            if level == 3:
                md_content.append(f"{indent}- **{label}**")
            else:
                md_content.append(f"{indent}- _{label}_")
        else:
            # Heading style with visual indicators
            if level <= 6:
                prefix = ""
                if level == 4:
                    prefix = "▸ "
                elif level == 5:
                    prefix = "▹ "
                elif level >= 6:
                    prefix = "◦ "

                md_content.append(f"{'#' * min(level, 6)} {prefix}{label}\n")
            else:
                # For levels deeper than 6, use bold with indentation
                indent = "  " * (level - 6)
                md_content.append(f"{indent}**◦ {label}**\n")

        # Add metadata
        self._add_concept_metadata_to_markdown(md_content, metadata,
                                               use_bullets)

    def _add_concept_metadata_to_markdown(self, md_content: List,
                                          metadata: Dict, use_bullets: bool):
        """Add concept metadata to markdown."""
        metadata_indent = "  " if use_bullets else ""

        if metadata['notation']:
            md_content.append(f"{metadata_indent}_Notation:_ "
                              f"`{metadata['notation']}`  ")
        if metadata['definition']:
            md_content.append(f"{metadata_indent}_Definition:_ "
                              f"{metadata['definition']}  ")
        if metadata['alt_labels']:
            md_content.append(f"{metadata_indent}_Alternative Labels:_ "
                              f"{', '.join(metadata['alt_labels'])}  ")

        # Add URI in smaller text
        md_content.append(f"{metadata_indent}<sub>URI: {metadata['uri']}</sub>\n")

    def _add_table_of_contents(self, md_content: List, schemes: Dict,
                               orphans_no_scheme: Set):
        """Add table of contents to markdown."""
        toc = ["## Table of Contents\n"]
        for scheme in sorted(schemes.keys(), key=lambda x: schemes[x]['label']):
            scheme_label = schemes[scheme]['label']
            anchor = scheme_label.lower().replace(' ', '-')
            toc.append(f"- [{scheme_label}](#concept-scheme-{anchor})")
        if orphans_no_scheme:
            toc.append("- [Unassigned Concepts](#unassigned-concepts)")
        toc.append("\n---\n")

        # Insert TOC after main title
        md_content[1:1] = toc

    def _process_schemes_to_markdown(self, schemes: Dict, md_content: List,
                                     add_concept_md, orphans_by_scheme: Dict):
        """Process schemes for markdown output."""
        for scheme in sorted(schemes.keys(), key=lambda x: schemes[x]['label']):
            scheme_data = schemes[scheme]
            md_content.append(f"## 📂 Concept Scheme: {scheme_data['label']}\n")
            md_content.append(f"<sub>URI: {scheme}</sub>\n")
            md_content.append("---\n")

            # Add top concepts in alphabetical order
            sorted_top_concepts = sorted(scheme_data['top_concepts'],
                                         key=self.get_label)
            for i, top_concept in enumerate(sorted_top_concepts):
                if i > 0:
                    md_content.append("")
                add_concept_md(top_concept, 3, use_bullets=False)

            # Add orphans that belong to this scheme
            if scheme in orphans_by_scheme and orphans_by_scheme[scheme]:
                md_content.append(f"\n### 📁 Other Concepts in "
                                  f"{scheme_data['label']}\n")
                md_content.append("_Concepts in this scheme without broader "
                                  "relations or top concept designation_\n")
                sorted_orphans = sorted(orphans_by_scheme[scheme],
                                        key=self.get_label)
                for orphan in sorted_orphans:
                    add_concept_md(orphan, 4, use_bullets=False)

    def _process_orphans_to_markdown(self, orphans_no_scheme: Set,
                                     md_content: List, add_concept_md):
        """Process orphan concepts for markdown output."""
        if orphans_no_scheme:
            md_content.append("\n## 📄 Unassigned Concepts\n")
            md_content.append("_Concepts not associated with any "
                              "concept scheme_\n")
            sorted_orphans = sorted(orphans_no_scheme, key=self.get_label)
            for orphan in sorted_orphans:
                add_concept_md(orphan, 3, use_bullets=False)

    def _write_markdown(self, output_file: str, md_content: List):
        """Write markdown file with formatting instructions."""
        final_content = []

        # Add custom formatting instructions at the top
        final_content.append("<!-- ")
        final_content.append("NOTION IMPORT TIPS:")
        final_content.append("1. Use Cmd/Ctrl+Shift+V to paste and preserve formatting")
        final_content.append("2. Convert to toggle lists: highlight text and press Cmd/Ctrl+Shift+7")
        final_content.append("3. Use synced blocks for concepts that appear in multiple places")
        final_content.append("-->\n")

        final_content.extend(md_content)

        # Write markdown
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(final_content))

    def to_notion_json(self, output_file: str):
        """Convert to JSON format that can be processed for Notion API."""
        schemes, hierarchy, _, orphans_by_scheme, orphans_no_scheme = \
            self.build_hierarchy()

        notion_data = {
            "vocabulary": {
                "schemes": [],
                "concepts": []
            }
        }

        processed = set()

        def build_concept_dict(concept, parent_id=None):
            """Build concept dictionary."""
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
        self._process_schemes_to_json(schemes, notion_data, build_concept_dict,
                                      orphans_by_scheme)

        # Add orphans with no scheme
        self._process_orphans_to_json(orphans_no_scheme, notion_data,
                                      build_concept_dict)

        # Write JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(notion_data, f, indent=2, ensure_ascii=False)

        logger.info("Created JSON file: %s", output_file)
        logger.info("Processed %d unique concepts", len(processed))

    def _process_schemes_to_json(self, schemes: Dict, notion_data: Dict,
                                 build_concept_dict, orphans_by_scheme: Dict):
        """Process schemes for JSON output."""
        for scheme in sorted(schemes.keys(), key=lambda x: schemes[x]['label']):
            scheme_data = schemes[scheme]
            scheme_id = str(scheme).replace('/', '_').replace('#', '_')
            scheme_dict = {
                "id": scheme_id,
                "title": scheme_data['label'],
                "uri": str(scheme),
                "top_concepts": [],
                "other_concepts": []
            }

            # Add top concepts in alphabetical order
            sorted_top_concepts = sorted(scheme_data['top_concepts'],
                                         key=self.get_label)
            for top_concept in sorted_top_concepts:
                concept_dict = build_concept_dict(top_concept, scheme_id)
                if concept_dict:
                    scheme_dict["top_concepts"].append(concept_dict)
                    notion_data["vocabulary"]["concepts"].append(concept_dict)

            # Add orphans that belong to this scheme
            if scheme in orphans_by_scheme and orphans_by_scheme[scheme]:
                sorted_orphans = sorted(orphans_by_scheme[scheme],
                                        key=self.get_label)
                for orphan in sorted_orphans:
                    concept_dict = build_concept_dict(orphan, scheme_id)
                    if concept_dict:
                        scheme_dict["other_concepts"].append(concept_dict)
                        notion_data["vocabulary"]["concepts"].append(concept_dict)

            notion_data["vocabulary"]["schemes"].append(scheme_dict)

    def _process_orphans_to_json(self, orphans_no_scheme: Set,
                                 notion_data: Dict, build_concept_dict):
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
                notion_data["vocabulary"]["unassigned_concepts"] = \
                    unassigned_concepts


class NotionToSKOSConverter:
    """Converter for Notion markdown to SKOS RDF format."""

    def __init__(self, namespace_uri: str = "http://example.org/vocabulary#",
                 prefix: str = "ex"):
        self.uri_manager = URIManager(namespace_uri)
        self.prefix = prefix
        self.graph = Graph()
        self.graph.bind(prefix, self.uri_manager.namespace)
        self.graph.bind('skos', SKOS)
        self.graph.bind('rdf', RDF)
        self.graph.bind('rdfs', RDFS)

    def parse_markdown(self, file_path: str) -> Graph:
        """Parse Notion markdown file and build SKOS graph."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            logger.error("Markdown file not found: %s", file_path)
            raise
        except PermissionError:
            logger.error("Permission denied reading file: %s", file_path)
            raise
        except UnicodeDecodeError as e:
            logger.error("File encoding issue - %s", e)
            logger.info("Try saving the file with UTF-8 encoding")
            raise
        except (IOError, OSError) as e:
            logger.error("Error reading markdown file: %s: %s",
                         type(e).__name__, e)
            raise

        current_scheme = None
        current_parent_stack = []
        i = 0

        logger.info("Parsing markdown file: %s", file_path)
        logger.info("Using namespace: %s", self.uri_manager.namespace_uri)
        logger.info("Using prefix: %s", self.prefix)

        try:
            while i < len(lines):
                i = self._process_line(lines, i, current_scheme,
                                       current_parent_stack)

        except (ValueError, TypeError, AttributeError) as e:
            logger.error("Error parsing markdown at line %d: %s",
                         i+1, type(e).__name__)
            logger.error("Details: %s", e)
            if i < len(lines):
                logger.error("Line %d: %s", i+1, lines[i].strip())
            raise

        logger.info("Parsed %d triples", len(self.graph))
        return self.graph

    def _process_line(self, lines: List[str], i: int, current_scheme: URIRef,
                      current_parent_stack: List) -> int:
        """Process a single line of markdown."""
        line = lines[i].strip()

        # Skip empty lines and comments
        if not line or line.startswith('<!--'):
            return i + 1

        # Skip table of contents
        if line == "## Table of Contents":
            while i < len(lines) and not lines[i].strip().startswith('#'):
                i += 1
            return i

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
                current_scheme = self._process_concept_scheme(title, metadata,
                                                              current_parent_stack)
            elif level >= 2 and current_scheme:
                self._process_concept(title, metadata, level, current_scheme,
                                      current_parent_stack)
            elif level >= 2 and not current_scheme:
                logger.warning("Found concept '%s' at line %d without a "
                               "concept scheme (H1)", title, i+1)
                logger.warning("Skipping this concept...")

        return i + 1

    def _clean_title(self, title: str) -> str:
        """Clean title by removing visual indicators."""
        # Remove visual indicators
        title = re.sub(r'^[▸▹◦📂📁📄]\s*', '', title)
        return title

    def _should_skip_section(self, title: str) -> bool:
        """Check if section should be skipped."""
        return (title.startswith('[') or
                title.startswith('Other Concepts') or
                title == 'Unassigned Concepts')

    def _extract_metadata(self, lines: List[str], start_index: int) -> Dict:
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
            if (metadata_line.startswith('_Definition:_') or
                    metadata_line.startswith('**Definition:**')):
                metadata['definition'] = metadata_line.split(':', 1)[1].strip()

            elif (metadata_line.startswith('_Alternative Labels:_') or
                  metadata_line.startswith('**Alternative Labels:**')):
                alt_text = metadata_line.split(':', 1)[1].strip()
                metadata['alt_labels'] = [label.strip()
                                          for label in alt_text.split(',')
                                          if label.strip()]

            elif (metadata_line.startswith('_Notation:_') or
                  metadata_line.startswith('**Notation:**')):
                metadata['notation'] = (metadata_line.split(':', 1)[1]
                                        .strip().strip('`'))

            elif (metadata_line.startswith('<sub>URI:') or
                  metadata_line.startswith('**URI:**')):
                uri_text = metadata_line
                uri_text = re.sub(r'<sub>URI:\s*|</sub>|URI:\s*|\*\*URI:\*\*\s*|`',
                                  '', uri_text).strip()
                if uri_text and uri_text != 'None':
                    metadata['existing_uri'] = uri_text

            j += 1

        return metadata

    def _process_concept_scheme(self, title: str, metadata: Dict,
                                current_parent_stack: List) -> URIRef:
        """Process a concept scheme (H1)."""
        if title.lower().startswith('concept scheme:'):
            title = title.split(':', 1)[1].strip()

        # Create or reuse URI
        scheme_uri = self.uri_manager.get_or_create_uri(title,
                                                        metadata['existing_uri'])

        # Add to graph
        self.graph.add((scheme_uri, RDF.type, SKOS.ConceptScheme))
        self.graph.add((scheme_uri, SKOS.prefLabel, Literal(title)))

        # Reset hierarchy tracking
        current_parent_stack.clear()
        current_parent_stack.append((1, scheme_uri, title))

        return scheme_uri

    def _process_concept(self, title: str, metadata: Dict, level: int,
                         current_scheme: URIRef, current_parent_stack: List):
        """Process a concept (H2+)."""
        # Create or reuse concept URI
        concept_uri = self.uri_manager.get_or_create_uri(title,
                                                         metadata['existing_uri'])

        # Add basic concept info
        self.graph.add((concept_uri, RDF.type, SKOS.Concept))
        self.graph.add((concept_uri, SKOS.prefLabel, Literal(title)))
        self.graph.add((concept_uri, SKOS.inScheme, current_scheme))

        # Add definition (or placeholder)
        if metadata['definition']:
            self.graph.add((concept_uri, SKOS.definition,
                            Literal(metadata['definition'])))
        else:
            self.graph.add((concept_uri, SKOS.definition,
                            Literal("Lorem ipsum")))

        # Add alternative labels
        for alt_label in metadata['alt_labels']:
            self.graph.add((concept_uri, SKOS.altLabel, Literal(alt_label)))

        # Add notation if present
        if metadata['notation']:
            self.graph.add((concept_uri, SKOS.notation,
                            Literal(metadata['notation'])))

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

    def export_turtle(self, output_file: str):
        """Export the graph as Turtle."""
        turtle_content = self.graph.serialize(format='turtle')

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(turtle_content)

        logger.info("Exported SKOS Turtle to: %s", output_file)
        logger.info("Total triples: %d", len(self.graph))

        # Count resources
        concepts = set(self.graph.subjects(RDF.type, SKOS.Concept))
        schemes = set(self.graph.subjects(RDF.type, SKOS.ConceptScheme))
        logger.info("Concept Schemes: %d", len(schemes))
        logger.info("Concepts: %d", len(concepts))

        # Show sample output
        logger.info("First few lines of output:")
        logger.info("=" * 50)
        lines = turtle_content.splitlines()
        for line in lines[:20]:
            logger.info(line)
        if len(lines) > 20:
            logger.info("...")
            logger.info("=" * 50)


def create_argument_parser() -> argparse.ArgumentParser:
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
    skos_parser = subparsers.add_parser('to-notion',
                                        help='Convert SKOS Turtle to Notion formats')
    skos_parser.add_argument('input_file', help='Input Turtle RDF file')
    skos_parser.add_argument('--format', choices=['csv', 'markdown', 'json', 'all'],
                             default='csv', help='Output format (default: csv)')
    skos_parser.add_argument('--output', help='Output file name (without extension)')
    skos_parser.add_argument('--skip-validation', action='store_true',
                             help='Skip SKOS validation checks')
    skos_parser.add_argument('--force', action='store_true',
                             help='Continue conversion even if validation finds errors')
    skos_parser.add_argument('--markdown-style',
                             choices=['headings', 'bullets', 'mixed'],
                             default='headings',
                             help='Markdown formatting style (default: headings)')
    skos_parser.add_argument('--language',
                             help='Preferred language for labels (e.g., en, fr, de)')
    skos_parser.add_argument('--fallback-languages', nargs='*',
                             help='Fallback languages in order of preference')
    skos_parser.add_argument('--batch-dir',
                             help='Process all .ttl files in directory')
    skos_parser.add_argument('--output-dir',
                             help='Output directory for batch processing')


def create_notion_to_skos_parser(subparsers):
    """Create parser for Notion to SKOS conversion."""
    notion_parser = subparsers.add_parser('to-skos',
                                          help='Convert Notion markdown to SKOS Turtle')
    notion_parser.add_argument('input_file', help='Input Notion markdown file')
    notion_parser.add_argument('--output', help='Output file name (default: input_skos.ttl)')
    notion_parser.add_argument('--namespace', default='http://example.org/vocabulary#',
                               help='Namespace URI for new concepts '
                                    '(default: http://example.org/vocabulary#)')
    notion_parser.add_argument('--prefix', default='ex',
                               help='Namespace prefix (default: ex)')
    notion_parser.add_argument('--batch-dir',
                               help='Process all .md files in directory')
    notion_parser.add_argument('--output-dir',
                               help='Output directory for batch processing')


def handle_to_notion_conversion(args) -> int:
    """Handle SKOS to Notion conversion."""
    logger.info("Processing SKOS to Notion conversion...")

    # Create configuration
    config = ConverterConfig()
    config.markdown_style = getattr(args, 'markdown_style', 'headings')

    # Set language preferences
    if hasattr(args, 'language') and args.language:
        fallbacks = getattr(args, 'fallback_languages', None) or ["en", ""]
        config.set_language_preferences(args.language, fallbacks)

    # Handle batch processing
    if hasattr(args, 'batch_dir') and args.batch_dir:
        if not args.output_dir:
            logger.error("--output-dir required for batch processing")
            return 1

        batch_processor = BatchProcessor(config)
        batch_processor.process_directory(args.batch_dir, args.output_dir,
                                          args.format, 'to-notion')
        return 0

    # Single file processing
    logger.info("Input file: %s", args.input_file)

    # Check if input file exists
    if not os.path.exists(args.input_file):
        logger.error("Input file not found: %s", args.input_file)
        logger.info("Current directory: %s", os.getcwd())
        return 1

    logger.info("Input file exists: %s", os.path.abspath(args.input_file))

    # Determine output base name
    if args.output:
        base_output = args.output
    else:
        base_output = args.input_file.rsplit('.', 1)[0] + '_notion'

    logger.info("Output base name: %s", base_output)

    # Create converter and load file
    converter = SKOSToNotionConverter(config)

    try:
        logger.info("Loading Turtle file...")
        converter.load_turtle(args.input_file)
    except (ValueError, TypeError, AttributeError, IOError, OSError):
        logger.error("Failed to load Turtle file")
        return 1

    # Run validation unless skipped
    if not args.skip_validation:
        is_valid = converter.validate_skos()

        if not is_valid and not args.force:
            logger.error("Validation found critical errors. Conversion aborted.")
            logger.info("Use --force to convert anyway, or fix the issues and try again.")
            logger.info("Use --skip-validation to skip validation entirely.")
            return 1
        elif not is_valid and args.force:
            logger.warning("Continuing with conversion despite errors...")

    # Perform conversion
    try:
        if args.format in ['csv', 'all']:
            converter.to_notion_csv(f"{base_output}.csv")

        if args.format in ['markdown', 'all']:
            converter.to_notion_markdown(f"{base_output}.md")

        if args.format in ['json', 'all']:
            converter.to_notion_json(f"{base_output}.json")
    except (ValueError, TypeError, AttributeError, IOError, OSError) as e:
        logger.error("Error during conversion: %s", type(e).__name__)
        logger.error("Details: %s", e)
        return 1

    print_notion_import_instructions()
    return 0


def handle_to_skos_conversion(args) -> int:
    """Handle Notion to SKOS conversion."""
    # Handle batch processing
    if hasattr(args, 'batch_dir') and args.batch_dir:
        if not args.output_dir:
            logger.error("--output-dir required for batch processing")
            return 1

        config = ConverterConfig()
        config.namespace_uri = args.namespace
        config.prefix = args.prefix

        batch_processor = BatchProcessor(config)
        batch_processor.process_directory(args.batch_dir, args.output_dir,
                                          '', 'to-skos')
        return 0

    # Single file processing
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
        logger.error("Failed to parse markdown file")
        return 1

    try:
        converter.export_turtle(output_file)
    except (IOError, OSError) as e:
        logger.error("Error exporting Turtle: %s", type(e).__name__)
        logger.error("Details: %s", e)
        return 1

    print_skos_conversion_summary()
    return 0


def configure_namespace(args) -> Tuple[str, str]:
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


def main():
    """Main entry point for the SKOS-Notion converter."""
    logger.info("Starting SKOS-Notion Converter...")

    try:
        parser = create_argument_parser()
        args = parser.parse_args()

        logger.info("Parsing command line arguments...")
        logger.debug("Parsed args: %s", args)

        if not args.command:
            logger.info("No command specified. Showing help...")
            parser.print_help()
            return 0

        if args.command == 'to-notion':
            return handle_to_notion_conversion(args)
        elif args.command == 'to-skos':
            return handle_to_skos_conversion(args)

    except KeyboardInterrupt:
        logger.warning("Conversion cancelled by user")
        return 1
    except (ValueError, TypeError, AttributeError, IOError, OSError) as e:
        logger.error("Unexpected error: %s", type(e).__name__)
        logger.error("Details: %s", e)
        logger.debug("Full traceback:", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    logger.info("Python version: %s", sys.version)
    logger.debug("Command line arguments: %s", sys.argv)

    exit_code = main()
    sys.exit(exit_code if exit_code else 0)