use crate::editor::compile::EvalCache;
use crate::editor::graph::{NodeGraph, NodeKey, NodeLayout, PortAddr, Side};
use crate::editor::registry::registry;
use crate::ui::text;
use crate::ui::theme::*;
use vello::Scene;
use vello::kurbo::{Affine, BezPath, Point, Rect, RoundedRect};
use vello::peniko::Color;

pub struct GraphCamera {
    pub pan: vello::kurbo::Vec2,
    pub zoom: f64,
}

impl GraphCamera {
    pub fn new() -> Self {
        Self {
            pan: vello::kurbo::Vec2::new(0.0, 0.0),
            zoom: 1.0,
        }
    }

    pub fn g2p(&self, p: Point) -> Point {
        Point::new(
            (p.x - self.pan.x) * self.zoom,
            (p.y - self.pan.y) * self.zoom,
        )
    }

    pub fn p2g(&self, p: Point) -> Point {
        Point::new(p.x / self.zoom + self.pan.x, p.y / self.zoom + self.pan.y)
    }

    pub fn affine(&self) -> Affine {
        Affine::scale(self.zoom) * Affine::translate(-self.pan)
    }
}

fn rounded_path(points: &[Point], radius: f64) -> BezPath {
    let mut path = BezPath::new();
    if points.is_empty() {
        return path;
    }
    path.move_to(points[0]);
    for i in 1..points.len() - 1 {
        let prev = points[i - 1];
        let curr = points[i];
        let next = points[i + 1];

        let d_prev = prev - curr;
        let len_prev = d_prev.hypot();
        let dir_prev = if len_prev > 0.0 {
            vello::kurbo::Vec2::new(d_prev.x / len_prev, d_prev.y / len_prev)
        } else {
            d_prev
        };

        let d_next = next - curr;
        let len_next = d_next.hypot();
        let dir_next = if len_next > 0.0 {
            vello::kurbo::Vec2::new(d_next.x / len_next, d_next.y / len_next)
        } else {
            d_next
        };

        let r = radius.min(len_prev / 2.0).min(len_next / 2.0);

        let start_arc = curr + dir_prev * r;
        let end_arc = curr + dir_next * r;

        path.line_to(start_arc);
        path.curve_to(
            curr + dir_prev * (r * 0.45),
            curr + dir_next * (r * 0.45),
            end_arc,
        );
    }
    path.line_to(*points.last().unwrap());
    path
}

pub struct NodeCanvas {
    pub camera: GraphCamera,
}

pub enum HitResult {
    None,
    NodeTitle(NodeKey),
    Socket(PortAddr),
}

impl NodeCanvas {
    pub fn new() -> Self {
        Self {
            camera: GraphCamera::new(),
        }
    }

    pub fn hit_test(&self, graph: &NodeGraph, g_pos: Point) -> HitResult {
        // Search in nodes (SlotMap doesn't preserve order, but that's fine for now)
        for (key, node) in graph.nodes.iter() {
            if let Some(layout) = &node.layout_cache {
                // Check sockets
                for (i, p) in layout.input_sockets.iter().enumerate() {
                    if p.distance(g_pos) <= SOCKET_HIT_R {
                        return HitResult::Socket(PortAddr {
                            node: key,
                            side: Side::In,
                            index: i as u16,
                        });
                    }
                }
                for (i, p) in layout.output_sockets.iter().enumerate() {
                    if p.distance(g_pos) <= SOCKET_HIT_R {
                        return HitResult::Socket(PortAddr {
                            node: key,
                            side: Side::Out,
                            index: i as u16,
                        });
                    }
                }

                // Check title bar
                let title_rect = Rect::from_origin_size(
                    node.pos,
                    vello::kurbo::Size::new(NODE_WIDTH, NODE_TITLE_H),
                );
                if title_rect.contains(g_pos) {
                    return HitResult::NodeTitle(key);
                }
            }
        }
        HitResult::None
    }

    pub fn socket_pos(&self, graph: &NodeGraph, addr: PortAddr) -> Option<Point> {
        let node = graph.nodes.get(addr.node)?;
        let layout = node.layout_cache.as_ref()?;
        if addr.side == Side::In {
            layout.input_sockets.get(addr.index as usize).copied()
        } else {
            layout.output_sockets.get(addr.index as usize).copied()
        }
    }

    pub fn draw(
        &mut self,
        graph: &mut NodeGraph,
        _cache: &EvalCache,
        panel_rect: Rect,
        scene: &mut Scene,
        temp_wire: Option<(PortAddr, Point)>,
    ) {
        let affine = self.camera.affine();
        let tl = self.camera.p2g(Point::new(0.0, 0.0));
        let br = self
            .camera
            .p2g(Point::new(panel_rect.width(), panel_rect.height()));

        // Draw grid
        let start_x = (tl.x / GRID_STEP).floor() * GRID_STEP;
        let start_y = (tl.y / GRID_STEP).floor() * GRID_STEP;

        for x in 0..=((br.x - start_x) / GRID_STEP).ceil() as i32 {
            let px = start_x + x as f64 * GRID_STEP;
            let path = vello::kurbo::Line::new(Point::new(px, tl.y), Point::new(px, br.y));
            scene.stroke(
                &vello::kurbo::Stroke::new(1.0 / self.camera.zoom),
                affine,
                COL_GRID,
                None,
                &path,
            );
        }
        for y in 0..=((br.y - start_y) / GRID_STEP).ceil() as i32 {
            let py = start_y + y as f64 * GRID_STEP;
            let path = vello::kurbo::Line::new(Point::new(tl.x, py), Point::new(br.x, py));
            scene.stroke(
                &vello::kurbo::Stroke::new(1.0 / self.camera.zoom),
                affine,
                COL_GRID,
                None,
                &path,
            );
        }

        // Layout nodes
        for (key, node) in &mut graph.nodes {
            if node.layout_cache.is_none() {
                let desc = registry().get(node.kind);
                let n_in = desc.inputs.len();
                let n_out = desc.outputs.len();
                let n_rows_sockets = n_in.max(n_out);
                let n_rows_widgets = if node.collapsed { 0 } else { desc.params.len() };

                let height = NODE_TITLE_H
                    + (n_rows_sockets as f64 * NODE_ROW_H)
                    + (n_rows_widgets as f64 * NODE_ROW_H)
                    + 8.0;
                let size = vello::kurbo::Size::new(NODE_WIDTH, height);

                let mut input_sockets = Vec::new();
                for i in 0..n_in {
                    let y = NODE_TITLE_H + (i as f64 + 0.5) * NODE_ROW_H;
                    input_sockets.push(node.pos + vello::kurbo::Vec2::new(0.0, y));
                }

                let mut output_sockets = Vec::new();
                for o in 0..n_out {
                    let y = NODE_TITLE_H + (o as f64 + 0.5) * NODE_ROW_H;
                    output_sockets.push(node.pos + vello::kurbo::Vec2::new(NODE_WIDTH, y));
                }

                node.layout_cache = Some(NodeLayout {
                    size,
                    title_rect: Rect::new(0.0, 0.0, NODE_WIDTH, NODE_TITLE_H),
                    input_sockets,
                    output_sockets,
                    widget_rows: vec![],
                });
            }
        }

        // Draw nodes
        for node in graph.nodes.values() {
            let layout = node.layout_cache.as_ref().unwrap();
            let rect = Rect::from_origin_size(node.pos, layout.size);
            let desc = registry().get(node.kind);

            scene.fill(
                vello::peniko::Fill::NonZero,
                affine,
                COL_NODE_BODY,
                None,
                &RoundedRect::from_rect(rect, NODE_CORNER),
            );

            let title_bg =
                Rect::from_origin_size(node.pos, vello::kurbo::Size::new(NODE_WIDTH, NODE_TITLE_H));
            // simplified title bg rect
            scene.fill(
                vello::peniko::Fill::NonZero,
                affine,
                category_color(desc.category),
                None,
                &title_bg,
            );
            let text_transform = affine * Affine::translate((node.pos.x + 8.0, node.pos.y + 4.0));
            text::draw_line(scene, text_transform, &node.title, 14.0, COL_TEXT);

            for (i, p) in layout.input_sockets.iter().enumerate() {
                let color = socket_color(desc.inputs[i].ty);
                scene.fill(
                    vello::peniko::Fill::NonZero,
                    affine,
                    color,
                    None,
                    &vello::kurbo::Circle::new(*p, SOCKET_R),
                );
            }

            for (i, p) in layout.output_sockets.iter().enumerate() {
                let color = socket_color(desc.outputs[i].ty);
                scene.fill(
                    vello::peniko::Fill::NonZero,
                    affine,
                    color,
                    None,
                    &vello::kurbo::Circle::new(*p, SOCKET_R),
                );
            }
        }

        let route_wire = |from_node: &crate::editor::graph::EditorNode,
                          from_layout: &NodeLayout,
                          from_p: Point,
                          to_node_rect: Rect,
                          to_p: Point|
         -> BezPath {
            let mut path = BezPath::new();
            if to_p.x < from_p.x + 40.0 {
                let safe_right_x = (from_p.x + 20.0).max(to_node_rect.max_x() + 20.0);
                let safe_left_x = (to_p.x - 20.0).min(from_node.pos.x - 20.0);
                let safe_bottom_y =
                    (from_node.pos.y + from_layout.size.height).max(to_node_rect.max_y()) + 20.0;
                let safe_top_y = from_node.pos.y.min(to_node_rect.min_y()) - 20.0;

                let dist_bottom = safe_bottom_y - from_p.y + safe_bottom_y - to_p.y;
                let dist_top = from_p.y - safe_top_y + to_p.y - safe_top_y;
                let safe_y = if dist_bottom < dist_top {
                    safe_bottom_y
                } else {
                    safe_top_y
                };

                let pts = [
                    from_p,
                    Point::new(safe_right_x, from_p.y),
                    Point::new(safe_right_x, safe_y),
                    Point::new(safe_left_x, safe_y),
                    Point::new(safe_left_x, to_p.y),
                    to_p,
                ];
                path = rounded_path(&pts, 20.0);
            } else {
                path.move_to(from_p);
                let dx = (to_p.x - from_p.x).abs().max(40.0) * 0.5;
                let c1 = Point::new(from_p.x + dx, from_p.y);
                let c2 = Point::new(to_p.x - dx, to_p.y);
                path.curve_to(c1, c2, to_p);
            }
            path
        };

        // Draw wires
        for edge in graph.edges.values() {
            let from_node = &graph.nodes[edge.from.node];
            let to_node = &graph.nodes[edge.to.node];
            let from_layout = from_node.layout_cache.as_ref().unwrap();
            let to_layout = to_node.layout_cache.as_ref().unwrap();

            let from_p = from_layout.output_sockets[edge.from.index as usize];
            let to_p = to_layout.input_sockets[edge.to.index as usize];
            let to_rect = Rect::from_origin_size(to_node.pos, to_layout.size);

            let path = route_wire(from_node, from_layout, from_p, to_rect, to_p);
            scene.stroke(
                &vello::kurbo::Stroke::new(WIRE_WIDTH),
                affine,
                COL_WIRE,
                None,
                &path,
            );
        }

        if let Some((from_addr, to_p)) = temp_wire {
            if let Some(from_node) = graph.nodes.get(from_addr.node) {
                if let Some(from_layout) = from_node.layout_cache.as_ref() {
                    let from_p = if from_addr.side == Side::In {
                        from_layout.input_sockets[from_addr.index as usize]
                    } else {
                        from_layout.output_sockets[from_addr.index as usize]
                    };

                    // For temp wire, if drawing from IN, reverse the logic slightly by swapping points
                    let path = if from_addr.side == Side::In {
                        // Drawing backwards from input socket
                        let from_rect = Rect::from_origin_size(from_node.pos, from_layout.size);
                        // We use to_p as "from" and from_p as "to" to reuse route_wire
                        // But we don't know the remote node rect, so we just pass a 0-sized rect at to_p
                        let to_rect =
                            Rect::from_origin_size(to_p, vello::kurbo::Size::new(0.0, 0.0));
                        // Since we are reversing the visual direction, the curve should flow the other way.
                        // Actually, route_wire assumes `from` is on the right and `to` is on the left.
                        // If we are dragging from an IN socket, the cursor is the `from_p` (source of data)
                        // and the socket is the `to_p` (destination of data).
                        route_wire(from_node, from_layout, to_p, from_rect, from_p)
                    } else {
                        // Drawing from an OUT socket
                        let to_rect =
                            Rect::from_origin_size(to_p, vello::kurbo::Size::new(0.0, 0.0));
                        route_wire(from_node, from_layout, from_p, to_rect, to_p)
                    };

                    scene.stroke(
                        &vello::kurbo::Stroke::new(WIRE_WIDTH),
                        affine,
                        COL_WIRE,
                        None,
                        &path,
                    );
                }
            }
        }
    }
}
