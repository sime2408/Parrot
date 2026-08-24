import SwiftUI

/// The report form. Everything that will be sent is on screen before it goes:
/// the screenshot as a thumbnail, the diagnostics verbatim. The buttons open a
/// pre-filled GitHub issue (or copy the text) — the user posts it, we never do.
struct BugReportSheet: View {
    /// Captured before the sheet appeared, so it shows the screen being
    /// reported rather than this form.
    let screenshot: NSImage?

    @Environment(\.dismiss) private var dismiss
    @State private var kind: BugReport.Kind = .bug
    @State private var title = ""
    @State private var details = ""
    @State private var includeScreenshot = true
    @State private var includeDiagnostics = true

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Report to the Scrapalot repo")
                .font(Theme.Typography.title())
                .foregroundStyle(Theme.Colors.ink)

            Picker("", selection: $kind) {
                ForEach(BugReport.Kind.allCases) { Text($0.label).tag($0) }
            }
            .pickerStyle(.segmented)
            .labelsHidden()

            TextField("Title", text: $title)
                .textFieldStyle(.roundedBorder)
                .font(Theme.Typography.body)

            VStack(alignment: .leading, spacing: 4) {
                TextEditor(text: $details)
                    .font(Theme.Typography.body)
                    .frame(height: 110)
                    .padding(4)
                    .overlay(RoundedRectangle(cornerRadius: Theme.Metrics.radius)
                        .strokeBorder(Theme.Colors.line))
                    .overlay(alignment: .topLeading) {
                        if details.isEmpty {
                            Text(kind.placeholder)
                                .font(Theme.Typography.body)
                                .foregroundStyle(Theme.Colors.ink3)
                                .padding(.horizontal, 9)
                                .padding(.vertical, 12)
                                .allowsHitTesting(false)
                        }
                    }
            }

            if let screenshot {
                Toggle(isOn: $includeScreenshot) {
                    HStack(spacing: 8) {
                        Image(nsImage: screenshot)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(width: 64, height: 40)
                            .clipShape(RoundedRectangle(cornerRadius: 4))
                            .overlay(RoundedRectangle(cornerRadius: 4)
                                .strokeBorder(Theme.Colors.line))
                        VStack(alignment: .leading, spacing: 1) {
                            Text("Attach a picture of this window")
                                .font(Theme.Typography.body)
                            Text("Check it doesn't show anything private — you'll paste it in yourself.")
                                .font(Theme.Typography.caption)
                                .foregroundStyle(Theme.Colors.ink2)
                        }
                    }
                }
            }

            Toggle(isOn: $includeDiagnostics) {
                VStack(alignment: .leading, spacing: 1) {
                    Text("Include version and settings")
                        .font(Theme.Typography.body)
                    Text(BugReport.diagnostics().joined(separator: "\n"))
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(Theme.Colors.ink2)
                        .textSelection(.enabled)
                }
            }

            Divider()

            HStack {
                Button("Cancel") { dismiss() }
                    .keyboardShortcut(.cancelAction)
                Spacer()
                Button("Copy as Text") {
                    BugReport.copyToPasteboard(text: "\(title)\n\n\(reportBody)")
                }
                Button("Open GitHub Issue") { submit() }
                    .keyboardShortcut(.defaultAction)
                    .disabled(title.trimmingCharacters(in: .whitespaces).isEmpty)
            }
        }
        .padding(Theme.Metrics.pad)
        .frame(width: 460)
    }

    private var reportBody: String {
        BugReport.body(text: details,
                       includeDiagnostics: includeDiagnostics,
                       includeScreenshot: includeScreenshot && screenshot != nil)
    }

    /// The screenshot goes on the clipboard last, so ⌘V in the browser pastes
    /// the picture and not something the user copied earlier.
    private func submit() {
        guard let url = BugReport.issueURL(kind: kind, title: title, body: reportBody) else { return }
        if includeScreenshot, let screenshot { BugReport.copyToPasteboard(screenshot) }
        NSWorkspace.shared.open(url)
        dismiss()
    }
}

/// The always-there corner button. Tinted rather than grey — the last quiet
/// circle we shipped went unnoticed — and it fills in on hover.
struct BugReportButton: View {
    let action: () -> Void
    @State private var hovering = false

    var body: some View {
        Button(action: action) {
            Image(systemName: "ladybug.fill")
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(hovering ? .white : Theme.Colors.accent)
                .frame(width: 30, height: 30)
                .background(hovering ? Theme.Colors.accent : Theme.Colors.accent.opacity(0.14),
                            in: Circle())
                .overlay(Circle().strokeBorder(Theme.Colors.accent.opacity(hovering ? 0 : 0.25)))
                .shadow(color: .black.opacity(hovering ? 0.18 : 0.08),
                        radius: hovering ? 6 : 3, y: 1)
                .scaleEffect(hovering ? 1.06 : 1)
        }
        .buttonStyle(.plain)
        .animation(.easeOut(duration: 0.12), value: hovering)
        .onHover { hovering = $0 }
        .help("Report a bug or suggest an idea")
        .accessibilityLabel("Report a bug or suggest an idea")
    }
}
