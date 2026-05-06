import { createOpencodeClient } from "@opencode-ai/sdk/v2/client";

/**
 * OpenCode client helper for OpenSwarm frontend
 * Usage after calling POST /api/opencode/start
 */
export function createOpenSwarmOpencodeClient(
  baseUrl: string,
  workspacePath: string,
  auth?: { username: string; password: string }
) {
  const headers: Record<string, string> = {};
  if (auth) {
    const token = btoa(`${auth.username}:${auth.password}`);
    headers["Authorization"] = `Basic ${token}`;
  }

  return createOpencodeClient({
    baseUrl,
    directory: workspacePath,
    // @ts-ignore - custom headers for basic auth
    headers,
  });
}

// Example usage in a React component (AgentChat or Dashboard)
// const client = createOpenSwarmOpencodeClient(
//   opencodeUrl,
//   workspacePath,
//   { username, password }
// );
// const session = await client.session.create({ title: "Swarm Agent" });
// await client.session.prompt({ sessionID: session.id, prompt: userMessage });
// client.event.subscribe(...) for real-time streaming
