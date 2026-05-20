//! Approval popup widget for TUI mode.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Clear, Paragraph, Wrap};

use crate::policy::{ApprovalDecision, ApprovalRequest};

/// Tracks pending approval state and renders the popup.
pub struct ApprovalState {
    request: Option<ApprovalRequest>,
}

impl ApprovalState {
    pub fn new() -> Self {
        Self { request: None }
    }

    /// Show the approval popup for the given request.
    pub fn show(&mut self, req: ApprovalRequest) {
        self.request = Some(req);
    }

    pub fn is_pending(&self) -> bool {
        self.request.is_some()
    }

    /// Try to handle a key press. Returns `Some(decision)` if a choice was
    /// made, `None` if the key was consumed but no decision yet.
    ///
    /// Keys 1/2/3 resolve immediately. All other keys are consumed while
    /// the popup is visible.
    pub fn handle_key(&mut self, ch: char) -> Option<ApprovalDecision> {
        let decision = match ch {
            '1' => ApprovalDecision::Allow,
            '2' => ApprovalDecision::AlwaysAllow,
            '3' => ApprovalDecision::Deny,
            _ => return None,
        };
        self.resolve(decision)
    }

    /// Resolve the pending approval with Esc → Deny.
    pub fn cancel(&mut self) {
        self.resolve(ApprovalDecision::Deny);
    }

    fn resolve(&mut self, decision: ApprovalDecision) -> Option<ApprovalDecision> {
        if let Some(req) = self.request.take() {
            req.response_tx.send(decision).ok();
            return Some(decision);
        }
        None
    }

    /// Render the approval popup as a floating overlay above the input area.
    pub fn render(&self, frame: &mut Frame, input_area: Rect) {
        let Some(req) = &self.request else { return };

        let width = (input_area.width).min(60);
        let height = 8u16; // border(2) + tool + reason + args + blank + options
        let popup_area =
            Rect { x: input_area.x + 1, y: input_area.y.saturating_sub(height), width, height };

        frame.render_widget(Clear, popup_area);

        let args_preview = truncate_str(&req.args, (width - 4) as usize - 6);

        let (border_color, title, reason_style, options) = if req.dangerous {
            (Color::Red, " DANGER ", Color::Red, "  [1] Allow   [3] Deny")
        } else {
            (
                Color::Yellow,
                " Approval ",
                Color::White,
                "  [1] Allow   [2] Always   [3] Deny",
            )
        };

        let lines = vec![
            Line::from(vec![
                Span::styled(" tool: ", Style::default().fg(border_color)),
                Span::styled(&req.tool_name, Style::default().fg(reason_style)),
            ]),
            Line::from(vec![
                Span::styled(" reason: ", Style::default().fg(border_color)),
                Span::styled(&req.reason, Style::default().fg(reason_style)),
            ]),
            Line::from(vec![
                Span::styled(" args: ", Style::default().fg(Color::DarkGray)),
                Span::styled(args_preview, Style::default().fg(Color::DarkGray)),
            ]),
            Line::from(""),
            Line::from(Span::styled(
                options,
                Style::default()
                    .fg(if req.dangerous { Color::Red } else { Color::Cyan })
                    .add_modifier(ratatui::style::Modifier::BOLD),
            )),
        ];

        let popup = Paragraph::new(lines)
            .block(
                Block::bordered()
                    .title(title)
                    .border_style(Style::default().fg(border_color)),
            )
            .wrap(Wrap { trim: true });
        frame.render_widget(popup, popup_area);
    }
}

fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_owned()
    } else {
        let end = s
            .char_indices()
            .take(max_len.saturating_sub(1))
            .last()
            .map(|(i, c)| i + c.len_utf8())
            .unwrap_or(0);
        format!("{}…", &s[..end])
    }
}
