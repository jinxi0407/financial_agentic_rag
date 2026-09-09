import { FormEvent, KeyboardEvent, useEffect, useRef } from "react";

interface ComposerProps {
  value: string;
  disabled: boolean;
  focusRequest: number;
  onChange: (value: string) => void;
  onSend: () => void;
}

export function Composer({ value, disabled, focusRequest, onChange, onSend }: ComposerProps) {
  const textarea = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    if (focusRequest) textarea.current?.focus();
  }, [focusRequest]);
  const submit = (event: FormEvent) => {
    event.preventDefault();
    onSend();
  };
  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      onSend();
    }
  };
  return (
    <form className="composer" onSubmit={submit}>
      <textarea
        ref={textarea}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={onKeyDown}
        placeholder="询问公司、财报、行情或新闻…"
        aria-label="输入金融研究问题"
        disabled={disabled}
        rows={1}
      />
      <button type="submit" aria-label="发送问题" disabled={disabled || !value.trim()}>
        ↑
      </button>
    </form>
  );
}
