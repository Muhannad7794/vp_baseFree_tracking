
// The code bellow is written in HLSL (High-Level Shading Language) inside a
// Custom Node within the Material Editor inside Unreal Engine.
// It is inserted here in the repo just for documentation purposes.

// --- SHADER 1: FRESNEL RIM (Ghost Effect) ---
// Calculate dot product between Normal and View Direction
float NdotV = dot(normalize(Normal), normalize(ViewDir));
// Invert and power it to create the "Rim" effect
float fresnel = pow(1.0 - saturate(NdotV), 3.0);

// --- SHADER 2: WORLD SPACE SCANLINES (Data Effect) ---
// Use World Position Z to create horizontal bars
// multiply by 'frequency' (50.0) and add Time
// alter the speed based on 'Sigma' (Tracking Instability)
float speed = 20.0 + (Sigma * 100.0);
float scanline = sin((WorldPos.z * 0.5) + (Time * speed));

// Sharpen the sine wave into thin lines
scanline = smoothstep(0.8, 1.0, scanline);

// --- COMBINATION ---
// Mix colors based on Stability (Sigma)
// If Sigma > 10 (Jittery), fade to Warning Color
float instability = saturate(Sigma / 10.0);
float3 finalColor = lerp(BaseColor, WarningColor, instability);

// Combine Scanline and Fresnel
// The scanline only appears "on top" of the fresnel glow
float finalAlpha = fresnel + scanline;

return float4(finalColor * finalAlpha, finalAlpha);