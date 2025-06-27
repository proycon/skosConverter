#!/usr/bin/env python3
"""
SKOS RDF to Notion Converter (and back!)
Converts SKOS vocabularies in Turtle format to CSV/Markdown for Notion import
Also converts Notion markdown back to SKOS Turtle format
"""

import csv
import json
import uuid
import re
from io import StringIO
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import SKOS, RDF, RDFS, DC, DCTERMS
import argparse
from collections import defaultdict


class SKOSToNotionConverter:
    def __init__(self):
        self.graph = Graph()
        self.skos = Namespace("http://www.w3.org/2004/02/skos/core#")
        
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
        except Exception as e:
            # Get more specific error information
            error_type = type(e).__name__
            error_msg = str(e)
            
            print(f"❌ Error parsing Turtle file: {error_type}")
            print(f"   Details: {error_msg}")
            
            # Try to extract line number from error if available
            line_match = re.search(r'line (\d+)', error_msg)
            if line_match:
                line_num = line_match.group(1)
                print(f"   Error at line: {line_num}")
                
                # Try to show the problematic line
                try:
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                        line_idx = int(line_num) - 1
                        if 0 <= line_idx < len(lines):
                            print(f"   Line {line_num}: {lines[line_idx].strip()}")
                            if line_idx > 0:
                                print(f"   Line {line_num-1}: {lines[line_idx-1].strip()}")
                            if line_idx < len(lines) - 1:
                                print(f"   Line {line_num+1}: {lines[line_idx+1].strip()}")
                except:
                    pass
            
            print("\n📌 Common Turtle syntax issues:")
            print("   - Missing '.' at the end of statements")
            print("   - Missing ';' between properties of the same subject")
            print("   - Unclosed brackets or quotes")
            print("   - Invalid URIs (missing < > brackets)")
            print("   - Invalid escape sequences in strings")
            print("   - Malformed prefixes or namespaces")
            print("\n💡 Tip: Try validating your Turtle file at: http://ttl.summerofcode.be/")
            raise
        
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
        
        # Build parent-child relationships - ensure each child appears only once
        children_assigned = set()
        
        for concept in all_concepts:
            # Get narrower concepts (children)
            for narrower in self.graph.objects(concept, SKOS.narrower):
                if narrower != concept and narrower not in children_assigned:  # Avoid self-reference and duplicates
                    hierarchy[concept].append(narrower)
                    children_assigned.add(narrower)
                
            # Get via broader relations (inverse) - only if not already assigned
            for child in self.graph.subjects(SKOS.broader, concept):
                if child != concept and child not in children_assigned and child not in hierarchy[concept]:
                    hierarchy[concept].append(child)
                    children_assigned.add(child)
        
        # Detect circular references
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
        
        # Find orphan concepts (not in hierarchy but are concepts)
        # A concept is an orphan if it has no broader concept and is not a top concept
        orph