## Ethical Use Guidelines

1. **Respectful Prompts**
   - Do not generate hateful, violent, or explicit imagery.
   - Avoid impersonating real individuals or creating misleading content.

2. **Content Filtering**
   - The application blocks obvious harmful keywords client-side.
   - Add stricter filtering if deploying publicly (e.g., integrate moderation APIs or classifier models).

3. **Transparency**
   - All images produced here are watermarked with “AI Generated • Demo”.
   - Preserve metadata (prompt, timestamp, parameters) when sharing outputs.

4. **Attribution & Licensing**
   - Stable Diffusion models are released under the CreativeML Open RAIL-M license.
   - Verify that downstream use complies with dataset and model licenses.

5. **User Safety**
   - Warn users about potential biases inherited from training data.
   - Provide reporting channels for inappropriate generations in production settings.

6. **Environmental Considerations**
   - Prefer GPU inference for energy efficiency.
   - Batch generations and reuse cached models to avoid redundant downloads.

