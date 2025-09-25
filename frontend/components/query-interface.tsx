"use client"

import type React from "react"

import { useState } from "react"
import { Send, Mic, ImageIcon } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"

interface QueryInterfaceProps {
  onSubmit: (query: string) => void
  isLoading?: boolean
}

export function QueryInterface({ onSubmit, isLoading = false }: QueryInterfaceProps) {
  const [query, setQuery] = useState("")

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (query.trim()) {
      onSubmit(query.trim())
      setQuery("")
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-3">
      <div className="relative">
        <Textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Describe what you're looking for..."
          className="min-h-[100px] pr-12 resize-none bg-white border-[#D8D8E5] text-[#030507] placeholder:text-[#858589]"
        />
        <div className="absolute bottom-3 right-3 flex gap-2">
          <Button type="button" size="sm" variant="ghost" className="p-2 h-8 w-8 text-[#575758] hover:text-[#030507]">
            <Mic className="w-4 h-4" />
          </Button>
          <Button type="button" size="sm" variant="ghost" className="p-2 h-8 w-8 text-[#575758] hover:text-[#030507]">
            <ImageIcon className="w-4 h-4" />
          </Button>
        </div>
      </div>

      <Button
        type="submit"
        className="w-full bg-[#030507] hover:bg-[#575758] text-white"
        disabled={isLoading || !query.trim()}
      >
        <Send className="w-4 h-4 mr-2" />
        {isLoading ? "Searching..." : "Search Products"}
      </Button>
    </form>
  )
}
