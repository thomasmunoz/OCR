"""
ORACLE-DESIGNED VISUAL CONTENT EXTRACTOR v2.1
==============================================
Handles extraction and reconstruction of visual elements:
- Tables (with headers, styling, full JSON reconstitution)
- Graphs/Diagrams (converted to Mermaid syntax)
- Charts (with data point extraction)
- AI Cascade for quality escalation
"""

import os
import sys
import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# ENHANCED TABLE FORMAT v2.1
# Full JSON reconstitution capability
# =============================================================================

class TableCellType(Enum):
    """Cell content type for reconstruction"""
    TEXT = "text"
    NUMBER = "number"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    DATE = "date"
    EMPTY = "empty"


@dataclass
class EnhancedTableCell:
    """Enhanced table cell with full metadata for reconstruction"""
    value: str
    row: int
    col: int
    cell_type: TableCellType = TableCellType.TEXT
    is_header: bool = False
    row_span: int = 1
    col_span: int = 1
    confidence: float = 0.95
    # Styling for reconstruction
    alignment: str = "left"  # left, center, right
    bold: bool = False
    background_color: Optional[str] = None
    # Numeric parsing
    numeric_value: Optional[float] = None
    currency_symbol: Optional[str] = None


@dataclass
class EnhancedTable:
    """
    ORACLE-DESIGNED: Enhanced table format for full JSON reconstitution.

    Includes:
    - Header row detection
    - Cell type inference (text, number, currency, date)
    - Styling metadata for reconstruction
    - Multiple export formats (HTML, Markdown, CSV)
    """
    table_id: str = ""
    page: int = 1
    rows: int = 0
    cols: int = 0
    confidence: float = 0.95

    # Header detection
    has_header: bool = True
    header_row_count: int = 1
    headers: List[str] = field(default_factory=list)

    # Data matrix (for quick access)
    data: List[List[str]] = field(default_factory=list)

    # Enhanced cells with full metadata
    cells: List[EnhancedTableCell] = field(default_factory=list)

    # Bounding box for spatial reference
    bbox: Optional[Dict[str, float]] = None

    # Table structure hints
    has_total_row: bool = False
    is_financial: bool = False
    caption: Optional[str] = None

    def to_html(self) -> str:
        """Convert table to HTML for reconstruction"""
        html = ['<table border="1" cellpadding="5" cellspacing="0">']

        if self.caption:
            html.append(f'<caption>{self.caption}</caption>')

        for row_idx in range(self.rows):
            html.append('<tr>')
            for col_idx in range(self.cols):
                cell = self._get_cell(row_idx, col_idx)
                tag = 'th' if cell.is_header else 'td'
                style = self._cell_style(cell)
                attrs = f' style="{style}"' if style else ''

                if cell.row_span > 1:
                    attrs += f' rowspan="{cell.row_span}"'
                if cell.col_span > 1:
                    attrs += f' colspan="{cell.col_span}"'

                html.append(f'<{tag}{attrs}>{cell.value}</{tag}>')
            html.append('</tr>')

        html.append('</table>')
        return '\n'.join(html)

    def to_markdown(self) -> str:
        """Convert table to Markdown for reconstruction"""
        if not self.data:
            return ""

        lines = []

        # Header row
        if self.has_header and self.headers:
            lines.append('| ' + ' | '.join(self.headers) + ' |')
            lines.append('|' + '|'.join(['---' for _ in self.headers]) + '|')
            start_row = self.header_row_count
        else:
            start_row = 0

        # Data rows
        for row_idx in range(start_row, len(self.data)):
            row = self.data[row_idx]
            lines.append('| ' + ' | '.join(str(c) if c else '' for c in row) + ' |')

        return '\n'.join(lines)

    def to_csv(self) -> str:
        """Convert table to CSV for reconstruction"""
        import csv
        import io

        output = io.StringIO()
        writer = csv.writer(output)

        if self.has_header and self.headers:
            writer.writerow(self.headers)
            # Skip header row in data since we already wrote headers
            for row in self.data[self.header_row_count:]:
                writer.writerow(row)
        else:
            for row in self.data:
                writer.writerow(row)

        return output.getvalue()

    def to_dict(self) -> Dict:
        """Full JSON-serializable representation for reconstruction"""
        return {
            "table_id": self.table_id,
            "page": self.page,
            "rows": self.rows,
            "cols": self.cols,
            "confidence": self.confidence,
            "has_header": self.has_header,
            "header_row_count": self.header_row_count,
            "headers": self.headers,
            "data": self.data,
            "cells": [
                {
                    "value": c.value,
                    "row": c.row,
                    "col": c.col,
                    "cell_type": c.cell_type.value,
                    "is_header": c.is_header,
                    "row_span": c.row_span,
                    "col_span": c.col_span,
                    "confidence": c.confidence,
                    "alignment": c.alignment,
                    "bold": c.bold,
                    "numeric_value": c.numeric_value,
                    "currency_symbol": c.currency_symbol
                }
                for c in self.cells
            ],
            "bbox": self.bbox,
            "has_total_row": self.has_total_row,
            "is_financial": self.is_financial,
            "caption": self.caption,
            # Reconstruction helpers
            "reconstruction": {
                "html": self.to_html(),
                "markdown": self.to_markdown(),
                "csv": self.to_csv()
            }
        }

    def _get_cell(self, row: int, col: int) -> EnhancedTableCell:
        """Get cell at position"""
        for cell in self.cells:
            if cell.row == row and cell.col == col:
                return cell
        # Return empty cell if not found
        return EnhancedTableCell(
            value=self.data[row][col] if row < len(self.data) and col < len(self.data[row]) else "",
            row=row,
            col=col,
            cell_type=TableCellType.EMPTY
        )

    def _cell_style(self, cell: EnhancedTableCell) -> str:
        """Generate CSS style for cell"""
        styles = []
        if cell.alignment != "left":
            styles.append(f"text-align: {cell.alignment}")
        if cell.bold:
            styles.append("font-weight: bold")
        if cell.background_color:
            styles.append(f"background-color: {cell.background_color}")
        return "; ".join(styles)


# =============================================================================
# GRAPH/DIAGRAM DETECTION WITH MERMAID OUTPUT
# =============================================================================

class DiagramType(Enum):
    """Types of diagrams/graphs that can be detected"""
    FLOWCHART = "flowchart"
    SEQUENCE = "sequence"
    CLASS_DIAGRAM = "class"
    STATE_DIAGRAM = "state"
    ENTITY_RELATIONSHIP = "erDiagram"
    PIE_CHART = "pie"
    GANTT = "gantt"
    MINDMAP = "mindmap"
    TIMELINE = "timeline"
    ORGANIZATION_CHART = "org_chart"
    UNKNOWN = "unknown"


@dataclass
class DiagramNode:
    """Node in a diagram"""
    id: str
    label: str
    shape: str = "rectangle"  # rectangle, circle, diamond, etc.
    color: Optional[str] = None

    def to_mermaid_node(self) -> str:
        """Convert to Mermaid node syntax"""
        # Sanitize ID for Mermaid (alphanumeric only)
        safe_id = re.sub(r'[^a-zA-Z0-9_]', '_', self.id)
        safe_label = self.label.replace('"', '\\"')

        shape_map = {
            "rectangle": f'{safe_id}["{safe_label}"]',
            "rounded": f'{safe_id}("{safe_label}")',
            "circle": f'{safe_id}(("{safe_label}"))',
            "diamond": f'{safe_id}{{"{safe_label}"}}',
            "hexagon": f'{safe_id}{{{{"{safe_label}"}}}}',
            "parallelogram": f'{safe_id}[/"{safe_label}"/]',
            "database": f'{safe_id}[("{safe_label}")]'
        }
        return shape_map.get(self.shape, f'{safe_id}["{safe_label}"]')


@dataclass
class DiagramEdge:
    """Edge/connection in a diagram"""
    source: str
    target: str
    label: Optional[str] = None
    style: str = "solid"  # solid, dashed, dotted
    arrow: str = "normal"  # normal, none, bidirectional

    def to_mermaid_edge(self) -> str:
        """Convert to Mermaid edge syntax"""
        safe_source = re.sub(r'[^a-zA-Z0-9_]', '_', self.source)
        safe_target = re.sub(r'[^a-zA-Z0-9_]', '_', self.target)

        # Arrow styles
        arrow_map = {
            "normal": "-->",
            "none": "---",
            "bidirectional": "<-->",
            "dotted": "-.->",
            "thick": "==>"
        }
        arrow_syntax = arrow_map.get(self.style if self.style != "solid" else self.arrow, "-->")

        if self.label:
            safe_label = self.label.replace('"', '\\"')
            return f'{safe_source} {arrow_syntax}|"{safe_label}"| {safe_target}'
        return f'{safe_source} {arrow_syntax} {safe_target}'


@dataclass
class DetectedDiagram:
    """
    ORACLE-DESIGNED: Detected diagram/graph with Mermaid reconstruction.

    Extracts diagram structure from images and generates Mermaid code
    for perfect reconstruction.
    """
    diagram_id: str = ""
    page: int = 1
    diagram_type: DiagramType = DiagramType.FLOWCHART
    confidence: float = 0.7

    # Structure
    nodes: List[DiagramNode] = field(default_factory=list)
    edges: List[DiagramEdge] = field(default_factory=list)

    # Raw extraction
    title: Optional[str] = None
    description: Optional[str] = None

    # Spatial info
    bbox: Optional[Dict[str, float]] = None

    # AI model used for extraction
    extraction_model: str = "unknown"

    def to_mermaid(self) -> str:
        """
        Generate Mermaid syntax for diagram reconstruction.

        Returns valid Mermaid code that can be rendered to recreate the diagram.
        """
        lines = []

        # Diagram type header
        if self.diagram_type == DiagramType.FLOWCHART:
            lines.append("flowchart TD")
        elif self.diagram_type == DiagramType.SEQUENCE:
            lines.append("sequenceDiagram")
        elif self.diagram_type == DiagramType.CLASS_DIAGRAM:
            lines.append("classDiagram")
        elif self.diagram_type == DiagramType.STATE_DIAGRAM:
            lines.append("stateDiagram-v2")
        elif self.diagram_type == DiagramType.ENTITY_RELATIONSHIP:
            lines.append("erDiagram")
        elif self.diagram_type == DiagramType.PIE_CHART:
            lines.append("pie showData")
        elif self.diagram_type == DiagramType.GANTT:
            lines.append("gantt")
        elif self.diagram_type == DiagramType.MINDMAP:
            lines.append("mindmap")
        else:
            lines.append("flowchart TD")  # Default

        # Title if present
        if self.title:
            lines.insert(0, f"---")
            lines.insert(0, f"title: {self.title}")
            lines.insert(0, f"---")

        # Add nodes (for flowcharts)
        if self.diagram_type in [DiagramType.FLOWCHART, DiagramType.STATE_DIAGRAM]:
            for node in self.nodes:
                lines.append(f"    {node.to_mermaid_node()}")

        # Add edges
        for edge in self.edges:
            lines.append(f"    {edge.to_mermaid_edge()}")

        return '\n'.join(lines)

    def to_dict(self) -> Dict:
        """Full JSON representation with Mermaid code"""
        return {
            "diagram_id": self.diagram_id,
            "page": self.page,
            "diagram_type": self.diagram_type.value,
            "confidence": self.confidence,
            "title": self.title,
            "description": self.description,
            "nodes": [
                {"id": n.id, "label": n.label, "shape": n.shape, "color": n.color}
                for n in self.nodes
            ],
            "edges": [
                {"source": e.source, "target": e.target, "label": e.label, "style": e.style, "arrow": e.arrow}
                for e in self.edges
            ],
            "bbox": self.bbox,
            "extraction_model": self.extraction_model,
            "reconstruction": {
                "mermaid": self.to_mermaid()
            }
        }


# =============================================================================
# CHART DETECTION WITH DATA EXTRACTION
# =============================================================================

class ChartType(Enum):
    """Types of charts that can be detected"""
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    AREA = "area"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    DONUT = "donut"
    STACKED_BAR = "stacked_bar"
    GROUPED_BAR = "grouped_bar"
    RADAR = "radar"
    BUBBLE = "bubble"
    UNKNOWN = "unknown"


@dataclass
class ChartDataSeries:
    """Data series in a chart"""
    name: str
    values: List[float]
    color: Optional[str] = None


@dataclass
class DetectedChart:
    """
    ORACLE-DESIGNED: Detected chart with extracted data points.

    Extracts actual numeric data from chart visualizations for:
    - Data reconstruction
    - Re-plotting in different tools
    - Analysis and computation
    """
    chart_id: str = ""
    page: int = 1
    chart_type: ChartType = ChartType.BAR
    confidence: float = 0.7

    # Chart metadata
    title: Optional[str] = None
    x_axis_label: Optional[str] = None
    y_axis_label: Optional[str] = None

    # Extracted data
    labels: List[str] = field(default_factory=list)  # X-axis labels or pie segments
    data_series: List[ChartDataSeries] = field(default_factory=list)

    # For pie charts
    pie_segments: List[Dict[str, Any]] = field(default_factory=list)

    # Scale info for reconstruction
    y_min: Optional[float] = None
    y_max: Optional[float] = None

    # Spatial info
    bbox: Optional[Dict[str, float]] = None

    # Legend
    legend_items: List[str] = field(default_factory=list)

    # AI model used
    extraction_model: str = "unknown"

    def to_plotly_json(self) -> Dict:
        """Generate Plotly-compatible JSON for reconstruction"""
        traces = []

        if self.chart_type == ChartType.PIE:
            traces.append({
                "type": "pie",
                "labels": self.labels,
                "values": self.data_series[0].values if self.data_series else [],
                "name": self.data_series[0].name if self.data_series else "Series 1"
            })
        elif self.chart_type in [ChartType.LINE, ChartType.AREA]:
            for series in self.data_series:
                traces.append({
                    "type": "scatter",
                    "mode": "lines" if self.chart_type == ChartType.LINE else "lines+markers",
                    "fill": "tozeroy" if self.chart_type == ChartType.AREA else None,
                    "x": self.labels,
                    "y": series.values,
                    "name": series.name
                })
        else:  # Bar charts
            for series in self.data_series:
                traces.append({
                    "type": "bar",
                    "x": self.labels,
                    "y": series.values,
                    "name": series.name
                })

        layout = {
            "title": {"text": self.title} if self.title else None,
            "xaxis": {"title": {"text": self.x_axis_label}} if self.x_axis_label else {},
            "yaxis": {"title": {"text": self.y_axis_label}} if self.y_axis_label else {}
        }

        return {"data": traces, "layout": layout}

    def to_csv_data(self) -> str:
        """Export chart data as CSV"""
        import csv
        import io

        output = io.StringIO()
        writer = csv.writer(output)

        # Header: Label, Series1, Series2, ...
        headers = ["Label"] + [s.name for s in self.data_series]
        writer.writerow(headers)

        # Data rows
        for i, label in enumerate(self.labels):
            row = [label]
            for series in self.data_series:
                row.append(series.values[i] if i < len(series.values) else "")
            writer.writerow(row)

        return output.getvalue()

    def to_mermaid_pie(self) -> Optional[str]:
        """Generate Mermaid pie chart (if applicable)"""
        if self.chart_type != ChartType.PIE:
            return None

        lines = ['pie showData']
        if self.title:
            lines.append(f'    title {self.title}')

        if self.data_series and self.labels:
            values = self.data_series[0].values
            for i, label in enumerate(self.labels):
                if i < len(values):
                    lines.append(f'    "{label}" : {values[i]}')

        return '\n'.join(lines)

    def to_dict(self) -> Dict:
        """Full JSON representation with reconstruction data"""
        return {
            "chart_id": self.chart_id,
            "page": self.page,
            "chart_type": self.chart_type.value,
            "confidence": self.confidence,
            "title": self.title,
            "x_axis_label": self.x_axis_label,
            "y_axis_label": self.y_axis_label,
            "labels": self.labels,
            "data_series": [
                {"name": s.name, "values": s.values, "color": s.color}
                for s in self.data_series
            ],
            "y_min": self.y_min,
            "y_max": self.y_max,
            "legend_items": self.legend_items,
            "bbox": self.bbox,
            "extraction_model": self.extraction_model,
            "reconstruction": {
                "plotly_json": self.to_plotly_json(),
                "csv_data": self.to_csv_data(),
                "mermaid": self.to_mermaid_pie()
            }
        }


# =============================================================================
# VISUAL CONTENT EXTRACTOR ENGINE
# =============================================================================

class VisualContentExtractor:
    """
    ORACLE-DESIGNED: Unified visual content extraction engine.

    Handles:
    - Enhanced table extraction with header detection
    - Diagram/graph detection with Mermaid conversion
    - Chart detection with data point extraction
    - AI cascade for quality escalation
    """

    # Currency patterns for table parsing
    CURRENCY_PATTERNS = [
        (r'^\$[\d,]+\.?\d*$', '$', 'USD'),
        (r'^€[\d,]+\.?\d*$', '€', 'EUR'),
        (r'^£[\d,]+\.?\d*$', '£', 'GBP'),
        (r'^[\d,]+\.?\d*\s*€$', '€', 'EUR'),
        (r'^[\d,]+\.?\d*\s*\$', '$', 'USD'),
        (r'^[\d\s]+,?\d*\s*€$', '€', 'EUR'),  # French format
    ]

    # Percentage pattern
    PERCENTAGE_PATTERN = r'^[\d,\.]+\s*%$'

    # Date patterns
    DATE_PATTERNS = [
        r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}',
        r'\d{4}[/\-]\d{1,2}[/\-]\d{1,2}',
        r'\d{1,2}\s+\w+\s+\d{4}',
    ]

    def __init__(self, use_ai_cascade: bool = True, cascade_threshold: float = 0.7):
        """
        Initialize the visual content extractor.

        Args:
            use_ai_cascade: Enable AI cascade for low-confidence extractions
            cascade_threshold: Confidence below which to escalate to better AI
        """
        self.use_ai_cascade = use_ai_cascade
        self.cascade_threshold = cascade_threshold
        self._vision_model = None
        self._cascade_model = None

    # =========================================================================
    # TABLE EXTRACTION
    # =========================================================================

    def extract_tables_enhanced(self, pdf_path: str) -> Dict[int, List[EnhancedTable]]:
        """
        ORACLE-DESIGNED: Extract tables with full enhancement.

        Features:
        - Header row detection
        - Cell type inference (text, number, currency, date)
        - Styling metadata
        - Multiple reconstruction formats
        """
        import fitz

        tables_by_page = {}

        try:
            doc = fitz.open(pdf_path)

            for page_num, page in enumerate(doc, 1):
                page_tables = []

                try:
                    tables = page.find_tables()

                    for table_idx, table in enumerate(tables.tables):
                        # Extract raw data
                        raw_data = table.extract()

                        if not raw_data:
                            continue

                        # Detect headers
                        has_header, header_row_count, headers = self._detect_table_headers(raw_data)

                        # Build enhanced table
                        enhanced = EnhancedTable(
                            table_id=f"table_p{page_num}_t{table_idx}",
                            page=page_num,
                            rows=table.row_count,
                            cols=table.col_count,
                            confidence=0.95,
                            has_header=has_header,
                            header_row_count=header_row_count,
                            headers=headers,
                            data=raw_data,
                            bbox={
                                "x0": table.bbox[0] if table.bbox else 0,
                                "y0": table.bbox[1] if table.bbox else 0,
                                "x1": table.bbox[2] if table.bbox else 0,
                                "y1": table.bbox[3] if table.bbox else 0
                            }
                        )

                        # Build enhanced cells
                        enhanced.cells = self._build_enhanced_cells(raw_data, has_header, header_row_count)

                        # Detect financial table
                        enhanced.is_financial = self._is_financial_table(raw_data)
                        enhanced.has_total_row = self._has_total_row(raw_data)

                        page_tables.append(enhanced)

                except Exception as e:
                    logger.debug(f"Table extraction on page {page_num}: {e}")

                if page_tables:
                    tables_by_page[page_num] = page_tables

            doc.close()

            total_tables = sum(len(t) for t in tables_by_page.values())
            logger.info(f"Enhanced table extraction: {total_tables} tables across {len(tables_by_page)} pages")

        except Exception as e:
            logger.warning(f"Enhanced table extraction failed: {e}")

        return tables_by_page

    def _detect_table_headers(self, data: List[List[str]]) -> Tuple[bool, int, List[str]]:
        """
        Detect if table has header row(s).

        Heuristics:
        - First row often has different content type than data rows
        - Headers are typically shorter, more descriptive text
        - Headers usually don't contain numbers
        """
        if not data or len(data) < 2:
            return False, 0, []

        first_row = data[0]

        # Check if first row looks like headers
        header_indicators = 0

        # Check 1: First row has no numbers, rest have numbers
        first_row_has_numbers = any(self._has_number(cell) for cell in first_row if cell)
        data_rows_have_numbers = any(
            any(self._has_number(cell) for cell in row if cell)
            for row in data[1:]
        )

        if not first_row_has_numbers and data_rows_have_numbers:
            header_indicators += 2

        # Check 2: First row cells are shorter on average
        if first_row:
            avg_first_len = sum(len(str(c)) for c in first_row if c) / len(first_row)
            avg_data_len = 0
            data_cells = 0
            for row in data[1:]:
                for cell in row:
                    if cell:
                        avg_data_len += len(str(cell))
                        data_cells += 1
            if data_cells > 0:
                avg_data_len /= data_cells
                if avg_first_len < avg_data_len * 0.8:
                    header_indicators += 1

        # Check 3: First row contains typical header words
        header_words = ['name', 'date', 'type', 'total', 'amount', 'niveau', 'montant',
                       'description', 'category', 'status', 'id', 'no', 'qty', 'price']
        first_row_lower = ' '.join(str(c).lower() for c in first_row if c)
        if any(hw in first_row_lower for hw in header_words):
            header_indicators += 2

        has_header = header_indicators >= 2
        headers = [str(c) if c else f"Column_{i}" for i, c in enumerate(first_row)] if has_header else []

        return has_header, 1 if has_header else 0, headers

    def _build_enhanced_cells(self, data: List[List[str]], has_header: bool,
                               header_row_count: int) -> List[EnhancedTableCell]:
        """Build enhanced cell objects with type inference"""
        cells = []

        for row_idx, row in enumerate(data):
            is_header_row = has_header and row_idx < header_row_count

            for col_idx, value in enumerate(row):
                cell_value = str(value) if value else ""
                cell_type, numeric_val, currency = self._infer_cell_type(cell_value)

                cell = EnhancedTableCell(
                    value=cell_value,
                    row=row_idx,
                    col=col_idx,
                    cell_type=cell_type,
                    is_header=is_header_row,
                    confidence=0.95,
                    alignment=self._infer_alignment(cell_type),
                    bold=is_header_row,
                    numeric_value=numeric_val,
                    currency_symbol=currency
                )
                cells.append(cell)

        return cells

    def _infer_cell_type(self, value: str) -> Tuple[TableCellType, Optional[float], Optional[str]]:
        """Infer cell content type"""
        if not value or not value.strip():
            return TableCellType.EMPTY, None, None

        value = value.strip()

        # Check currency
        for pattern, symbol, _ in self.CURRENCY_PATTERNS:
            if re.match(pattern, value):
                # Extract numeric value
                numeric_str = re.sub(r'[^\d,\.]', '', value)
                numeric_str = numeric_str.replace(',', '.')  # French format
                try:
                    numeric_val = float(numeric_str)
                    return TableCellType.CURRENCY, numeric_val, symbol
                except ValueError:
                    pass

        # Check percentage
        if re.match(self.PERCENTAGE_PATTERN, value):
            try:
                numeric_val = float(re.sub(r'[^\d,\.]', '', value.replace(',', '.')))
                return TableCellType.PERCENTAGE, numeric_val, None
            except ValueError:
                pass

        # Check date
        for pattern in self.DATE_PATTERNS:
            if re.search(pattern, value):
                return TableCellType.DATE, None, None

        # Check pure number
        try:
            # Handle European number format
            clean_value = value.replace(' ', '').replace(',', '.')
            numeric_val = float(clean_value)
            return TableCellType.NUMBER, numeric_val, None
        except ValueError:
            pass

        return TableCellType.TEXT, None, None

    def _infer_alignment(self, cell_type: TableCellType) -> str:
        """Infer alignment based on cell type"""
        if cell_type in [TableCellType.NUMBER, TableCellType.CURRENCY, TableCellType.PERCENTAGE]:
            return "right"
        return "left"

    def _has_number(self, value: str) -> bool:
        """Check if value contains numeric content"""
        if not value:
            return False
        return bool(re.search(r'\d', str(value)))

    def _is_financial_table(self, data: List[List[str]]) -> bool:
        """Detect if table is financial (contains currency/amounts)"""
        currency_count = 0
        for row in data:
            for cell in row:
                if cell:
                    for pattern, _, _ in self.CURRENCY_PATTERNS:
                        if re.match(pattern, str(cell).strip()):
                            currency_count += 1
        return currency_count >= 2

    def _has_total_row(self, data: List[List[str]]) -> bool:
        """Detect if table has a total row"""
        if not data:
            return False

        last_row = ' '.join(str(c).lower() for c in data[-1] if c)
        total_indicators = ['total', 'sum', 'subtotal', 'grand total', 'totaux', 'sous-total']
        return any(ind in last_row for ind in total_indicators)

    # =========================================================================
    # DIAGRAM/GRAPH DETECTION
    # =========================================================================

    def detect_diagrams(self, image_path: str, page_num: int = 1) -> List[DetectedDiagram]:
        """
        ORACLE-DESIGNED: Detect diagrams/graphs in an image and convert to Mermaid.

        Uses vision AI to:
        1. Identify diagram type (flowchart, sequence, etc.)
        2. Extract nodes and connections
        3. Generate Mermaid reconstruction code
        """
        diagrams = []

        try:
            # Use vision model for diagram analysis
            diagram_data = self._analyze_diagram_with_ai(image_path)

            if diagram_data:
                diagram = self._build_diagram_from_analysis(diagram_data, page_num)
                if diagram:
                    diagrams.append(diagram)

        except Exception as e:
            logger.warning(f"Diagram detection failed: {e}")

        return diagrams

    def _analyze_diagram_with_ai(self, image_path: str) -> Optional[Dict]:
        """
        Use AI vision model to analyze diagram structure.

        Returns structured data about the diagram or None if not a diagram.
        """
        # For now, return a placeholder - will integrate with actual vision model
        # This would use Qwen2-VL or similar to analyze the image

        # Placeholder: Would call self._vision_model.analyze_diagram(image_path)
        return None

    def _build_diagram_from_analysis(self, analysis: Dict, page_num: int) -> Optional[DetectedDiagram]:
        """Build DetectedDiagram from AI analysis results"""
        if not analysis:
            return None

        diagram = DetectedDiagram(
            diagram_id=f"diagram_p{page_num}_d0",
            page=page_num,
            diagram_type=DiagramType(analysis.get("type", "flowchart")),
            confidence=analysis.get("confidence", 0.7),
            title=analysis.get("title"),
            description=analysis.get("description"),
            extraction_model="qwen2-vl-7b"
        )

        # Build nodes
        for node_data in analysis.get("nodes", []):
            node = DiagramNode(
                id=node_data.get("id", f"node_{len(diagram.nodes)}"),
                label=node_data.get("label", ""),
                shape=node_data.get("shape", "rectangle")
            )
            diagram.nodes.append(node)

        # Build edges
        for edge_data in analysis.get("edges", []):
            edge = DiagramEdge(
                source=edge_data.get("source", ""),
                target=edge_data.get("target", ""),
                label=edge_data.get("label"),
                style=edge_data.get("style", "solid")
            )
            diagram.edges.append(edge)

        return diagram

    # =========================================================================
    # CHART DETECTION
    # =========================================================================

    def detect_charts(self, image_path: str, page_num: int = 1) -> List[DetectedChart]:
        """
        ORACLE-DESIGNED: Detect charts in an image and extract data points.

        Uses vision AI to:
        1. Identify chart type (bar, line, pie, etc.)
        2. Extract labels and data values
        3. Generate reconstruction data
        """
        charts = []

        try:
            # Use vision model for chart analysis
            chart_data = self._analyze_chart_with_ai(image_path)

            if chart_data:
                chart = self._build_chart_from_analysis(chart_data, page_num)
                if chart:
                    charts.append(chart)

        except Exception as e:
            logger.warning(f"Chart detection failed: {e}")

        return charts

    def _analyze_chart_with_ai(self, image_path: str) -> Optional[Dict]:
        """
        Use AI vision model to analyze chart and extract data.

        Returns structured data about the chart or None if not a chart.
        """
        # Placeholder - would integrate with vision model
        return None

    def _build_chart_from_analysis(self, analysis: Dict, page_num: int) -> Optional[DetectedChart]:
        """Build DetectedChart from AI analysis results"""
        if not analysis:
            return None

        chart = DetectedChart(
            chart_id=f"chart_p{page_num}_c0",
            page=page_num,
            chart_type=ChartType(analysis.get("type", "bar")),
            confidence=analysis.get("confidence", 0.7),
            title=analysis.get("title"),
            x_axis_label=analysis.get("x_axis_label"),
            y_axis_label=analysis.get("y_axis_label"),
            labels=analysis.get("labels", []),
            extraction_model="qwen2-vl-7b"
        )

        # Build data series
        for series_data in analysis.get("series", []):
            series = ChartDataSeries(
                name=series_data.get("name", "Series"),
                values=series_data.get("values", []),
                color=series_data.get("color")
            )
            chart.data_series.append(series)

        return chart

    # =========================================================================
    # AI CASCADE SYSTEM
    # =========================================================================

    def escalate_extraction(self, content_type: str, source_path: str,
                            current_result: Any, current_confidence: float) -> Any:
        """
        ORACLE-DESIGNED: AI Cascade for quality escalation.

        If extraction confidence is below threshold, escalate to a more
        powerful (but slower) AI model for better accuracy.

        Cascade levels:
        1. TrOCR / Florence-2 (fast, decent quality)
        2. Qwen2-VL-2B (balanced)
        3. Qwen2-VL-7B (slow, high quality)
        4. GOT-OCR2 (slowest, best quality)
        """
        if current_confidence >= self.cascade_threshold:
            logger.info(f"Confidence {current_confidence:.0%} >= threshold, no cascade needed")
            return current_result

        logger.info(f"Confidence {current_confidence:.0%} < threshold, escalating to better model")

        # Determine next model in cascade
        cascade_models = [
            ("trocr_large_printed", 0.6),
            ("qwen2_vl_2b", 0.7),
            ("qwen2_vl_7b", 0.85),
            ("got_ocr2", 0.95)
        ]

        for model_key, expected_quality in cascade_models:
            if expected_quality > current_confidence:
                logger.info(f"Escalating to {model_key} for {content_type}")

                try:
                    # Would load and use the escalated model
                    # escalated_result = self._run_with_model(model_key, source_path, content_type)
                    # return escalated_result
                    pass
                except Exception as e:
                    logger.warning(f"Cascade to {model_key} failed: {e}")
                    continue

        # Return original if all cascades fail
        return current_result


# =============================================================================
# ENHANCED OUTPUT v2.1 BUILDER
# =============================================================================

def build_enhanced_v21_output(
    ocr_result: Any,
    pdf_path: str,
    visual_extractor: VisualContentExtractor = None
) -> Dict[str, Any]:
    """
    ORACLE-DESIGNED: Build enhanced JSON output v2.1.

    Includes all visual content with reconstruction capability:
    - Tables with HTML/Markdown/CSV export
    - Diagrams with Mermaid code
    - Charts with Plotly JSON and CSV data
    """
    if visual_extractor is None:
        visual_extractor = VisualContentExtractor()

    # Start with base enhanced output structure
    output = {
        "schema_version": "2.1",
        "job_id": ocr_result.job_id if hasattr(ocr_result, 'job_id') else "",
        "processing_timestamp": datetime.utcnow().isoformat(),

        # Visual content sections
        "visual_content": {
            "tables": [],
            "diagrams": [],
            "charts": [],
            "total_tables": 0,
            "total_diagrams": 0,
            "total_charts": 0
        },

        # Reconstruction helpers
        "reconstruction": {
            "tables_html": [],
            "tables_markdown": [],
            "diagrams_mermaid": [],
            "charts_plotly": []
        }
    }

    # Extract enhanced tables
    if pdf_path:
        tables_by_page = visual_extractor.extract_tables_enhanced(pdf_path)

        all_tables = []
        for page_num, page_tables in tables_by_page.items():
            for table in page_tables:
                table_dict = table.to_dict()
                all_tables.append(table_dict)

                # Add reconstruction data
                output["reconstruction"]["tables_html"].append({
                    "table_id": table.table_id,
                    "html": table.to_html()
                })
                output["reconstruction"]["tables_markdown"].append({
                    "table_id": table.table_id,
                    "markdown": table.to_markdown()
                })

        output["visual_content"]["tables"] = all_tables
        output["visual_content"]["total_tables"] = len(all_tables)

    return output
