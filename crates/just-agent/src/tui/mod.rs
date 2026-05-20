//! TUI interface for interactive agent mode.

mod approval;
mod completion;
mod history;
mod markdown;
mod wrap;

use std::time::Instant;

use ratatui::Frame;
use ratatui::crossterm::event::{
    KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseEvent, MouseEventKind,
};
use ratatui::layout::{Constraint, Layout, Margin};
use ratatui::style::{Color, Style, Stylize};
use ratatui::text::{Line, Text};
use ratatui::widgets::{Block, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState, Wrap};
use ratatui_textarea::TextArea;
use tokio::sync::mpsc;

use crate::command::{self, SlashCommand, UserInput};
use crate::policy::ApprovalRequest;
use crate::types::AgentEvent;
use approval::ApprovalState;
use completion::CompletionState;
use wrap::word_wrap_line_count;

/// A line in the chat output area.
#[derive(Debug)]
pub enum ChatLine {
    User(String),
    Assistant(String),
    ToolCall { name: String, args: String },
    ToolResult(String),
    Reasoning(String),
    Status(String),
    Error(String),
    System(String),
}

/// TUI application state.
pub struct App {
    pub chat_lines: Vec<ChatLine>,
    pub textarea: TextArea<'static>,
    pub auto_scroll: bool,
    pub agent_busy: bool,
    pub should_quit: bool,
    completion: CompletionState,
    approval: ApprovalState,
    history: history::InputHistory,
    scroll_pos: usize,
    content_length: usize,
    visible_height: usize,
}

impl App {
    pub fn new() -> Self {
        let mut textarea = TextArea::default();
        textarea.set_block(
            Block::bordered()
                .title(">> ")
                .border_style(Style::default().fg(Color::DarkGray)),
        );
        textarea.set_placeholder_text("Type a message...");
        Self {
            chat_lines: Vec::new(),
            textarea,
            scroll_pos: 0,
            content_length: 0,
            visible_height: 0,
            auto_scroll: true,
            agent_busy: false,
            should_quit: false,
            completion: CompletionState::new(),
            approval: ApprovalState::new(),
            history: history::InputHistory::new(),
        }
    }

    /// Render the TUI.
    pub fn render(&mut self, frame: &mut Frame) {
        let [chat_area, input_area] =
            Layout::vertical([Constraint::Fill(1), Constraint::Length(5)]).areas(frame.area());

        let auto_scroll = self.auto_scroll;
        let old_pos = self.scroll_pos;

        let t0 = Instant::now();
        let text = self.build_chat_text(chat_area.width);
        let build_ms = t0.elapsed().as_millis();

        let content_width = chat_area.width.saturating_sub(2) as usize;
        let visible_height = chat_area.height.saturating_sub(2) as usize;
        let t1 = Instant::now();
        let total = word_wrap_line_count(&text, content_width);
        let wrap_ms = t1.elapsed().as_millis();

        if build_ms + wrap_ms > 3 {
            tracing::warn!(
                "render: build={}ms wrap={}ms lines={}",
                build_ms,
                wrap_ms,
                total
            );
        }

        let pos = if auto_scroll {
            total.saturating_sub(visible_height)
        } else {
            old_pos.min(total.saturating_sub(visible_height))
        };

        let paragraph = Paragraph::new(text)
            .block(Block::bordered().title("Chat"))
            .wrap(Wrap { trim: true })
            .scroll((pos as u16, 0));
        frame.render_widget(paragraph, chat_area);

        // Scrollbar, only when content overflows viewport.
        //
        // Ratatui calculates thumb position as:
        //   thumb_end = (position + viewport) * track / (content_length - 1 + viewport)
        //
        // If content_length = total (all wrapped lines), at max scroll the denominator
        // is (total - 1 + viewport), which is larger than (position + viewport) = total,
        // so the thumb never reaches the track bottom.
        //
        // Setting content_length = scroll_range + 1 makes the denominator equal total,
        // so the thumb reaches the track bottom at max scroll.
        let scroll_range = total.saturating_sub(visible_height);
        if scroll_range > 0 {
            let mut scrollbar_state = ScrollbarState::new(scroll_range + 1)
                .position(pos)
                .viewport_content_length(visible_height);
            frame.render_stateful_widget(
                Scrollbar::new(ScrollbarOrientation::VerticalRight),
                chat_area.inner(Margin { vertical: 1, horizontal: 0 }),
                &mut scrollbar_state,
            );
        }

        self.scroll_pos = pos;
        self.content_length = total;
        self.visible_height = visible_height;

        self.completion.render(frame, input_area);
        self.approval.render(frame, input_area);
        frame.render_widget(&self.textarea, input_area);
    }

    /// Show an approval request popup.
    pub fn show_approval(&mut self, req: ApprovalRequest) {
        self.approval.show(req);
    }

    /// Handle a crossterm key event.
    pub fn handle_key_event(&mut self, key: KeyEvent, prompt_tx: &mpsc::Sender<UserInput>) {
        if key.kind != KeyEventKind::Press {
            return;
        }

        // Approval popup: intercept 1/2/3 keys
        if self.approval.is_pending() {
            if let KeyCode::Char(ch) = key.code {
                self.approval.handle_key(ch);
            } else if key.code == KeyCode::Esc {
                self.approval.cancel();
            }
            return;
        }
        if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
            if self.completion.is_visible() {
                self.completion.hide();
            }
            return;
        }

        // Scroll keys
        match key.code {
            KeyCode::PageUp => {
                self.scroll_pos = self.scroll_pos.saturating_sub(10);
                self.auto_scroll = false;
                return;
            }
            KeyCode::PageDown => {
                self.scroll_pos = self.scroll_pos.saturating_add(10);
                self.auto_scroll = false;
                return;
            }
            _ => {}
        }

        // History navigation (when completion popup is not visible)
        if !self.completion.is_visible() {
            match key.code {
                KeyCode::Up => {
                    let current = self.textarea.lines().join("\n");
                    if let Some(entry) = self.history.up(&current) {
                        self.textarea.clear();
                        self.textarea.insert_str(entry);
                    }
                    return;
                }
                KeyCode::Down => {
                    if let Some(result) = self.history.down() {
                        self.textarea.clear();
                        match result {
                            history::Either::Entry(s) => {
                                self.textarea.insert_str(s);
                            }
                            history::Either::Draft(s) => {
                                self.textarea.insert_str(s);
                            }
                        }
                    }
                    return;
                }
                _ => {}
            }
        }

        // Completion popup navigation
        if self.completion.is_visible() {
            match key.code {
                KeyCode::Up => {
                    self.completion.move_up();
                    return;
                }
                KeyCode::Down => {
                    self.completion.move_down();
                    return;
                }
                KeyCode::Tab => {
                    if let Some(cmd) = self.completion.selected_command() {
                        self.textarea.clear();
                        self.textarea.insert_str(cmd.name);
                        self.textarea.insert_char(' ');
                        self.completion.hide();
                        return;
                    }
                }
                KeyCode::Esc => {
                    self.completion.hide();
                    return;
                }
                _ => {}
            }
        }

        // Enter submits input (unless Shift is held)
        if key.code == KeyCode::Enter
            && !key
                .modifiers
                .intersects(KeyModifiers::SHIFT | KeyModifiers::CONTROL)
        {
            // If completion popup is visible, resolve to selected candidate first
            if self.completion.is_visible() {
                if let Some(cmd) = self.completion.selected_command() {
                    self.textarea.clear();
                    self.textarea.insert_str(cmd.name);
                }
                self.completion.hide();
            }

            let text = self.textarea.lines().join("\n");
            if !text.is_empty() && !self.agent_busy {
                self.auto_scroll = true;
                self.history.push(text.clone());
                self.textarea.clear();
                self.completion.hide();

                match command::parse(&text) {
                    None => {
                        self.chat_lines.push(ChatLine::User(text.clone()));
                        prompt_tx.try_send(UserInput::Prompt(text)).ok();
                    }
                    Some(Ok(cmd)) => {
                        self.dispatch_command(cmd, prompt_tx);
                    }
                    Some(Err(msg)) => {
                        self.chat_lines.push(ChatLine::Error(msg));
                    }
                }
            }
            return;
        }

        // Forward all other keys to textarea, then update completion
        self.textarea.input(key);
        let text = self.textarea.lines().join("\n");
        self.completion.update(&text);
    }

    /// Handle an event from the agent task.
    pub fn handle_agent_event(&mut self, event: AgentEvent) {
        match event {
            AgentEvent::Reasoning(text) => {
                self.chat_lines.push(ChatLine::Reasoning(text));
                self.auto_scroll = true;
            }
            AgentEvent::AssistantContent(text) => {
                self.chat_lines.push(ChatLine::Assistant(text));
                self.auto_scroll = true;
            }
            AgentEvent::ToolCall { name, args } => {
                self.chat_lines.push(ChatLine::ToolCall { name, args });
                self.auto_scroll = true;
            }
            AgentEvent::ToolResult(result) => {
                self.chat_lines.push(ChatLine::ToolResult(result));
                self.auto_scroll = true;
            }
            AgentEvent::Finished(content) => {
                self.chat_lines.push(ChatLine::Assistant(content));
                self.agent_busy = false;
                self.auto_scroll = true;
            }
            AgentEvent::MaxRoundsExceeded => {
                self.chat_lines
                    .push(ChatLine::Error("max rounds exceeded".into()));
                self.agent_busy = false;
                self.auto_scroll = true;
            }
            AgentEvent::Error(err) => {
                self.chat_lines.push(ChatLine::Error(err));
                self.agent_busy = false;
                self.auto_scroll = true;
            }
            AgentEvent::Status(msg) => {
                self.chat_lines.push(ChatLine::Status(msg));
                self.auto_scroll = true;
            }
            AgentEvent::Busy => {
                self.agent_busy = true;
            }
        }
    }

    /// Handle a mouse scroll event in the chat area.
    pub fn handle_mouse_event(&mut self, event: MouseEvent, _chat_area_height: u16) {
        match event.kind {
            MouseEventKind::ScrollUp => {
                self.scroll_pos = self.scroll_pos.saturating_sub(3);
                self.auto_scroll = false;
            }
            MouseEventKind::ScrollDown => {
                self.scroll_pos = self.scroll_pos.saturating_add(3);
                // Re-enable auto_scroll if scrolled to bottom
                let max_pos = self.content_length.saturating_sub(self.visible_height);
                if self.scroll_pos >= max_pos {
                    self.auto_scroll = true;
                }
            }
            _ => {}
        }
    }

    /// Dispatch a parsed slash command.
    fn dispatch_command(&mut self, cmd: SlashCommand, prompt_tx: &mpsc::Sender<UserInput>) {
        match cmd {
            SlashCommand::Help => {
                self.chat_lines.push(ChatLine::System(command::help_text()));
                self.auto_scroll = true;
            }
            SlashCommand::Quit => {
                self.should_quit = true;
            }
            SlashCommand::Clear => {
                self.chat_lines.clear();
            }
            SlashCommand::Status => {
                prompt_tx
                    .try_send(UserInput::Command(SlashCommand::Status))
                    .ok();
                self.chat_lines
                    .push(ChatLine::System("requesting status...".into()));
                self.auto_scroll = true;
            }
            SlashCommand::Compact => {
                prompt_tx
                    .try_send(UserInput::Command(SlashCommand::Compact))
                    .ok();
                self.chat_lines
                    .push(ChatLine::System("running compaction...".into()));
                self.auto_scroll = true;
            }
            SlashCommand::Skill { name } => {
                prompt_tx
                    .try_send(UserInput::Command(SlashCommand::Skill {
                        name: name.clone(),
                    }))
                    .ok();
                self.chat_lines
                    .push(ChatLine::System(format!("loading skill: {name}...")));
                self.auto_scroll = true;
            }
        }
    }

    /// Build styled Text from chat_lines for rendering.
    fn build_chat_text(&self, term_width: u16) -> Text<'_> {
        let mut lines: Vec<Line> = Vec::new();
        for entry in &self.chat_lines {
            match entry {
                ChatLine::User(text) => {
                    lines.push(Line::from(vec![
                        ">> ".bold().fg(Color::Green),
                        text.clone().into(),
                    ]));
                }
                ChatLine::Assistant(text) => {
                    lines.extend(markdown::render_markdown(text, term_width));
                }
                ChatLine::ToolCall { name, args } => {
                    lines.push(Line::from(vec![
                        "[tool] ".dim().fg(Color::Yellow),
                        format!("{name}({args})").dim(),
                    ]));
                }
                ChatLine::ToolResult(result) => {
                    lines.push(Line::from(vec![
                        "[result] ".dim().fg(Color::Cyan),
                        result.clone().dim(),
                    ]));
                }
                ChatLine::Reasoning(text) => {
                    lines.push(Line::from(vec![
                        "[think] ".dim().fg(Color::Magenta),
                        text.clone().italic().dim(),
                    ]));
                }
                ChatLine::Status(msg) => {
                    lines.push(Line::from(msg.clone().dim().italic()));
                }
                ChatLine::Error(err) => {
                    lines.push(Line::from(vec![
                        "[error] ".fg(Color::Red),
                        err.clone().fg(Color::Red),
                    ]));
                }
                ChatLine::System(msg) => {
                    for (i, line) in msg.lines().enumerate() {
                        let prefix = if i == 0 { "[system] " } else { "          " };
                        lines.push(Line::from(vec![
                            prefix.fg(Color::DarkGray),
                            line.to_owned().fg(Color::DarkGray),
                        ]));
                    }
                }
            }
        }
        Text::from(lines)
    }
}
