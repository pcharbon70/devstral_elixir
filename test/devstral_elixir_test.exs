defmodule DevstralElixirTest do
  use ExUnit.Case
  doctest DevstralElixir

  test "greets the world" do
    assert DevstralElixir.hello() == :world
  end
end
