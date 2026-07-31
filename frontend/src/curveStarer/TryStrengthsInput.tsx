export function TryStrengthsInput({
  value,
  onChange,
  disabled,
  className,
}: {
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
  className?: string;
}) {
  return (
    <label className={`cr-try-strengths-input${className ? ` ${className}` : ""}`}>
      <span className="cr-try-strengths-input__label">Try Strengths：</span>
      <input
        type="text"
        className="cr-try-strengths-input__field"
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
        spellCheck={false}
        aria-label="Try strengths"
        placeholder="{0.0001, 0.00001, 0.000001}"
      />
    </label>
  );
}
